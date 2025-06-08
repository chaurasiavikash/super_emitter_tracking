# ============================================================================
# FILE: src/analysis/emission_quantifier.py
# ============================================================================
import numpy as np
import pandas as pd
import xarray as xr
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from scipy import integrate, optimize
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class EmissionQuantificationResult:
    """Result of emission quantification analysis."""
    emitter_id: str
    emission_rate_kg_hr: float
    emission_rate_uncertainty_kg_hr: float
    emission_rate_kg_day: float
    emission_rate_tonnes_year: float
    flux_kg_m2_s: float
    method_used: str
    confidence_level: float
    quality_flags: Dict[str, bool]
    meteorological_data: Dict[str, float]
    spatial_data: Dict[str, float]

class EmissionQuantifier:
    """
    Quantify methane emission rates from super-emitter detections.
    
    Features:
    - Multiple quantification methods (mass balance, plume fitting, inversion)
    - Uncertainty estimation and error propagation
    - Integration with meteorological data
    - Quality assessment and validation
    - Multiple output units and temporal aggregations
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.quantification_config = config.get('analysis', {}).get('emission_quantification', {})
        
        # Physical constants
        self.MOLECULAR_WEIGHT_CH4 = 16.04  # g/mol
        self.MOLECULAR_WEIGHT_AIR = 28.97  # g/mol
        self.AVOGADRO = 6.022e23  # molecules/mol
        self.GAS_CONSTANT = 8.314  # J/(mol·K)
        self.STANDARD_PRESSURE = 101325.0  # Pa
        self.STANDARD_TEMPERATURE = 273.15  # K
        
        # Default quantification parameters
        self.default_wind_speed = 5.0  # m/s
        self.default_boundary_layer_height = 1000.0  # m
        self.default_mixing_efficiency = 0.5  # dimensionless
        
        logger.info("EmissionQuantifier initialized")
    
    def quantify_emissions(self, detections: pd.DataFrame,
                          enhancement_data: Optional[xr.Dataset] = None,
                          meteorological_data: Optional[xr.Dataset] = None,
                          method: str = 'mass_balance') -> List[EmissionQuantificationResult]:
        """
        Quantify emission rates for detected super-emitters.
        
        Args:
            detections: DataFrame with super-emitter detections
            enhancement_data: xarray Dataset with methane enhancement maps
            meteorological_data: xarray Dataset with wind and boundary layer data
            method: Quantification method ('mass_balance', 'gaussian_plume', 'inversion')
            
        Returns:
            List of emission quantification results
        """
        
        logger.info(f"Quantifying emissions for {len(detections)} detections using {method} method")
        
        if len(detections) == 0:
            return []
        
        results = []
        
        for idx, detection in detections.iterrows():
            try:
                # Get meteorological data for this detection
                met_data = self._extract_meteorological_data(detection, meteorological_data)
                
                # Get enhancement data for this detection
                enh_data = self._extract_enhancement_data(detection, enhancement_data)
                
                # Quantify emission based on method
                if method == 'mass_balance':
                    result = self._mass_balance_quantification(detection, enh_data, met_data)
                elif method == 'gaussian_plume':
                    result = self._gaussian_plume_quantification(detection, enh_data, met_data)
                elif method == 'inversion':
                    result = self._inversion_quantification(detection, enh_data, met_data)
                else:
                    logger.warning(f"Unknown quantification method: {method}")
                    continue
                
                if result:
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Emission quantification failed for detection {idx}: {e}")
                continue
        
        logger.info(f"Successfully quantified emissions for {len(results)} detections")
        return results
    
    def _mass_balance_quantification(self, detection: pd.Series,
                                   enhancement_data: Dict,
                                   met_data: Dict) -> Optional[EmissionQuantificationResult]:
        """
        Quantify emissions using mass balance approach.
        
        This is based on the simple box model: Q = C × U × H × W
        where Q = emission rate, C = concentration enhancement, 
        U = wind speed, H = boundary layer height, W = plume width
        """
        
        # Extract key parameters
        enhancement_ppb = detection.get('mean_enhancement', 0)
        area_km2 = detection.get('spatial_extent_km2', 1.0)
        
        # Get meteorological parameters
        wind_speed = met_data.get('wind_speed', self.default_wind_speed)
        boundary_layer_height = met_data.get('boundary_layer_height', self.default_boundary_layer_height)
        temperature = met_data.get('temperature', 288.15)  # K
        pressure = met_data.get('pressure', self.STANDARD_PRESSURE)  # Pa
        
        # Convert enhancement to mass concentration
        mass_concentration = self._ppb_to_kg_m3(enhancement_ppb, temperature, pressure)
        
        # Estimate plume dimensions
        plume_width = np.sqrt(area_km2 * 1e6)  # Convert km2 to m and assume square
        plume_height = min(boundary_layer_height, 500.0)  # Assume plume doesn't fill entire BL
        
        # Mass balance calculation
        # Q = C × U × A_cross_section
        cross_sectional_area = plume_width * plume_height  # m²
        emission_rate_kg_s = mass_concentration * wind_speed * cross_sectional_area
        
        # Convert to kg/hr
        emission_rate_kg_hr = emission_rate_kg_s * 3600
        
        # Estimate uncertainty (simplified approach)
        # In practice, this would use proper error propagation
        relative_uncertainty = np.sqrt(
            (0.3)**2 +  # 30% uncertainty in concentration
            (0.2)**2 +  # 20% uncertainty in wind speed
            (0.4)**2    # 40% uncertainty in plume dimensions
        )
        
        uncertainty_kg_hr = emission_rate_kg_hr * relative_uncertainty
        
        # Quality assessment
        quality_flags = self._assess_quantification_quality(
            detection, enhancement_data, met_data, 'mass_balance'
        )
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(quality_flags, met_data)
        
        return EmissionQuantificationResult(
            emitter_id=detection.get('emitter_id', f"EMIT_{detection.name}"),
            emission_rate_kg_hr=emission_rate_kg_hr,
            emission_rate_uncertainty_kg_hr=uncertainty_kg_hr,
            emission_rate_kg_day=emission_rate_kg_hr * 24,
            emission_rate_tonnes_year=emission_rate_kg_hr * 24 * 365.25 / 1000,
            flux_kg_m2_s=emission_rate_kg_s / (area_km2 * 1e6),
            method_used='mass_balance',
            confidence_level=confidence_level,
            quality_flags=quality_flags,
            meteorological_data=met_data,
            spatial_data={
                'area_km2': area_km2,
                'plume_width_m': plume_width,
                'plume_height_m': plume_height
            }
        )
    
    def _gaussian_plume_quantification(self, detection: pd.Series,
                                     enhancement_data: Dict,
                                     met_data: Dict) -> Optional[EmissionQuantificationResult]:
        """
        Quantify emissions using Gaussian plume model fitting.
        
        Fits a Gaussian plume model to the enhancement pattern to estimate source strength.
        """
        
        # This requires 2D enhancement data around the source
        if not enhancement_data or 'enhancement_2d' not in enhancement_data:
            logger.warning("Gaussian plume fitting requires 2D enhancement data")
            return None
        
        enhancement_2d = enhancement_data['enhancement_2d']
        lats = enhancement_data.get('lats', [])
        lons = enhancement_data.get('lons', [])
        
        if len(lats) == 0 or len(lons) == 0:
            return None
        
        # Convert coordinates to distances (simplified)
        source_lat = detection.get('center_lat', np.mean(lats))
        source_lon = detection.get('center_lon', np.mean(lons))
        
        # Calculate distances from source
        lat_dist = (lats - source_lat) * 111320  # m
        lon_dist = (lons - source_lon) * 111320 * np.cos(np.radians(source_lat))  # m
        
        X, Y = np.meshgrid(lon_dist, lat_dist)
        
        # Get meteorological parameters
        wind_speed = met_data.get('wind_speed', self.default_wind_speed)
        wind_direction = met_data.get('wind_direction', 0)  # degrees
        stability_class = met_data.get('stability_class', 'D')  # Pasquill stability class
        
        # Define Gaussian plume function
        def gaussian_plume(coords, Q, x0, y0, sigma_y, sigma_z, H):
            """Gaussian plume model."""
            x, y = coords
            
            # Rotate coordinates based on wind direction
            wind_rad = np.radians(wind_direction)
            x_rot = (x - x0) * np.cos(wind_rad) + (y - y0) * np.sin(wind_rad)
            y_rot = -(x - x0) * np.sin(wind_rad) + (y - y0) * np.cos(wind_rad)
            
            # Only consider downwind points
            x_rot = np.maximum(x_rot, 1.0)  # Avoid division by zero
            
            # Gaussian plume equation (ground-level concentration)
            concentration = (Q / (2 * np.pi * wind_speed * sigma_y * sigma_z)) * \
                          np.exp(-0.5 * (y_rot / sigma_y)**2) * \
                          np.exp(-0.5 * (H / sigma_z)**2)
            
            return concentration
        
        # Estimate dispersion parameters based on stability class
        sigma_y, sigma_z = self._estimate_dispersion_parameters(X, stability_class)
        
        try:
            # Flatten data for fitting
            coords = np.vstack([X.ravel(), Y.ravel()])
            enhancement_flat = enhancement_2d.ravel()
            
            # Remove NaN values
            valid_mask = ~np.isnan(enhancement_flat)
            coords_valid = coords[:, valid_mask]
            enhancement_valid = enhancement_flat[valid_mask]
            
            if len(enhancement_valid) < 10:
                logger.warning("Insufficient valid data for Gaussian plume fitting")
                return None
            
            # Initial parameter guess
            initial_guess = [
                1000.0,  # Q (emission rate, arbitrary units)
                0.0,     # x0 (source x position)
                0.0,     # y0 (source y position)
                np.std(coords_valid[1]) * 0.5,  # sigma_y
                100.0,   # sigma_z
                0.0      # H (effective height)
            ]
            
            # Fit Gaussian plume model
            from scipy.optimize import curve_fit
            
            popt, pcov = curve_fit(
                lambda coords, Q, x0, y0, sigma_y, sigma_z, H: 
                gaussian_plume(coords, Q, x0, y0, sigma_y, sigma_z, H),
                coords_valid,
                enhancement_valid,
                p0=initial_guess,
                maxfev=1000
            )
            
            # Extract fitted parameters
            Q_fit, x0_fit, y0_fit, sigma_y_fit, sigma_z_fit, H_fit = popt
            
            # Convert from concentration to emission rate
            # The fitted Q is in concentration units, need to convert to mass flow
            temperature = met_data.get('temperature', 288.15)
            pressure = met_data.get('pressure', self.STANDARD_PRESSURE)
            
            # Approximate conversion (simplified)
            emission_rate_kg_s = abs(Q_fit) * wind_speed * sigma_y_fit * sigma_z_fit * \
                               self._ppb_to_kg_m3(1.0, temperature, pressure) / 1e-9
            
            emission_rate_kg_hr = emission_rate_kg_s * 3600
            
            # Estimate uncertainty from covariance matrix
            Q_uncertainty = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else Q_fit * 0.5
            relative_uncertainty = Q_uncertainty / abs(Q_fit)
            uncertainty_kg_hr = emission_rate_kg_hr * relative_uncertainty
            
            # Quality assessment
            quality_flags = self._assess_quantification_quality(
                detection, enhancement_data, met_data, 'gaussian_plume'
            )
            quality_flags['plume_fit_quality'] = relative_uncertainty < 1.0
            
            confidence_level = self._calculate_confidence_level(quality_flags, met_data)
            
            return EmissionQuantificationResult(
                emitter_id=detection.get('emitter_id', f"EMIT_{detection.name}"),
                emission_rate_kg_hr=emission_rate_kg_hr,
                emission_rate_uncertainty_kg_hr=uncertainty_kg_hr,
                emission_rate_kg_day=emission_rate_kg_hr * 24,
                emission_rate_tonnes_year=emission_rate_kg_hr * 24 * 365.25 / 1000,
                flux_kg_m2_s=emission_rate_kg_s / (detection.get('spatial_extent_km2', 1.0) * 1e6),
                method_used='gaussian_plume',
                confidence_level=confidence_level,
                quality_flags=quality_flags,
                meteorological_data=met_data,
                spatial_data={
                    'fitted_source_x': x0_fit,
                    'fitted_source_y': y0_fit,
                    'sigma_y': sigma_y_fit,
                    'sigma_z': sigma_z_fit,
                    'effective_height': H_fit
                }
            )
            
        except Exception as e:
            logger.error(f"Gaussian plume fitting failed: {e}")
            return None
    
    def _inversion_quantification(self, detection: pd.Series,
                                enhancement_data: Dict,
                                met_data: Dict) -> Optional[EmissionQuantificationResult]:
        """
        Quantify emissions using atmospheric inversion approach.
        
        This is a placeholder for more sophisticated inversion methods
        that would use atmospheric transport models.
        """
        
        logger.warning("Atmospheric inversion method not fully implemented")
        
        # For now, fall back to mass balance method
        return self._mass_balance_quantification(detection, enhancement_data, met_data)
    
    def _extract_meteorological_data(self, detection: pd.Series,
                                   meteorological_data: Optional[xr.Dataset]) -> Dict:
        """Extract meteorological data for a specific detection."""
        
        if meteorological_data is None:
            logger.warning("No meteorological data provided, using defaults")
            return {
                'wind_speed': self.default_wind_speed,
                'wind_direction': 0.0,
                'boundary_layer_height': self.default_boundary_layer_height,
                'temperature': 288.15,
                'pressure': self.STANDARD_PRESSURE,
                'stability_class': 'D'
            }
        
        # Extract data at detection location and time
        lat = detection.get('center_lat')
        lon = detection.get('center_lon')
        timestamp = detection.get('timestamp', datetime.now())
        
        try:
            # Interpolate meteorological data to detection location
            met_point = meteorological_data.interp(
                lat=lat, lon=lon, time=timestamp, method='linear'
            )
            
            return {
                'wind_speed': float(met_point.get('wind_speed', self.default_wind_speed)),
                'wind_direction': float(met_point.get('wind_direction', 0.0)),
                'boundary_layer_height': float(met_point.get('boundary_layer_height', self.default_boundary_layer_height)),
                'temperature': float(met_point.get('temperature_2m', 288.15)),
                'pressure': float(met_point.get('surface_pressure', self.STANDARD_PRESSURE)),
                'stability_class': 'D'  # Would need to calculate from other variables
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract meteorological data: {e}")
            return {
                'wind_speed': self.default_wind_speed,
                'wind_direction': 0.0,
                'boundary_layer_height': self.default_boundary_layer_height,
                'temperature': 288.15,
                'pressure': self.STANDARD_PRESSURE,
                'stability_class': 'D'
            }
    
    def _extract_enhancement_data(self, detection: pd.Series,
                                enhancement_data: Optional[xr.Dataset]) -> Dict:
        """Extract enhancement data around a specific detection."""
        
        if enhancement_data is None:
            return {'enhancement_value': detection.get('mean_enhancement', 0)}
        
        lat = detection.get('center_lat')
        lon = detection.get('center_lon')
        
        try:
            # Extract local enhancement data around the detection
            # Define a window around the detection point
            lat_window = 0.1  # degrees
            lon_window = 0.1  # degrees
            
            # Select region around detection
            local_data = enhancement_data.sel(
                lat=slice(lat - lat_window, lat + lat_window),
                lon=slice(lon - lon_window, lon + lon_window)
            )
            
            if 'enhancement' in local_data.data_vars:
                enhancement_2d = local_data.enhancement.values
                lats = local_data.lat.values
                lons = local_data.lon.values
                
                return {
                    'enhancement_2d': enhancement_2d,
                    'lats': lats,
                    'lons': lons,
                    'enhancement_value': detection.get('mean_enhancement', np.nanmean(enhancement_2d))
                }
            
        except Exception as e:
            logger.warning(f"Failed to extract enhancement data: {e}")
        
        return {'enhancement_value': detection.get('mean_enhancement', 0)}
    
    def _ppb_to_kg_m3(self, concentration_ppb: float, temperature: float, pressure: float) -> float:
        """Convert methane concentration from ppb to kg/m³."""
        
        # Convert ppb to mole fraction
        mole_fraction = concentration_ppb * 1e-9
        
        # Calculate air density using ideal gas law
        air_density = (pressure * self.MOLECULAR_WEIGHT_AIR / 1000) / (self.GAS_CONSTANT * temperature)  # kg/m³
        
        # Calculate CH4 mass concentration
        ch4_concentration = mole_fraction * air_density * (self.MOLECULAR_WEIGHT_CH4 / self.MOLECULAR_WEIGHT_AIR)
        
        return ch4_concentration
    
    def _estimate_dispersion_parameters(self, distance_grid: np.ndarray, stability_class: str) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate Pasquill-Gifford dispersion parameters."""
        
        # Simplified dispersion parameter estimation
        # In practice, would use proper Pasquill-Gifford curves
        
        distance = np.abs(distance_grid)
        distance = np.maximum(distance, 100.0)  # Minimum distance
        
        if stability_class in ['A', 'B']:  # Very unstable, unstable
            sigma_y = 0.32 * distance * (1 + 0.0004 * distance)**(-0.5)
            sigma_z = 0.24 * distance * (1 + 0.001 * distance)**(-0.5)
        elif stability_class in ['C', 'D']:  # Slightly unstable, neutral
            sigma_y = 0.22 * distance * (1 + 0.0004 * distance)**(-0.5)
            sigma_z = 0.20 * distance
        else:  # E, F - stable, very stable
            sigma_y = 0.16 * distance * (1 + 0.0004 * distance)**(-0.5)
            sigma_z = 0.14 * distance * (1 + 0.0003 * distance)**(-0.5)
        
        return sigma_y, sigma_z
    
    def _assess_quantification_quality(self, detection: pd.Series,
                                     enhancement_data: Dict,
                                     met_data: Dict,
                                     method: str) -> Dict[str, bool]:
        """Assess quality of emission quantification."""
        
        quality_flags = {}
        
        # Check meteorological data quality
        quality_flags['has_wind_data'] = 'wind_speed' in met_data and met_data['wind_speed'] > 0
        quality_flags['realistic_wind_speed'] = met_data.get('wind_speed', 0) > 1.0 and met_data.get('wind_speed', 0) < 20.0
        quality_flags['has_boundary_layer_data'] = 'boundary_layer_height' in met_data
        
        # Check enhancement data quality
        enhancement_value = enhancement_data.get('enhancement_value', detection.get('mean_enhancement', 0))
        quality_flags['significant_enhancement'] = enhancement_value > 10.0  # ppb
        quality_flags['realistic_enhancement'] = enhancement_value < 1000.0  # ppb
        
        # Check detection quality
        quality_flags['high_detection_score'] = detection.get('detection_score', 0) > 0.7
        quality_flags['reasonable_area'] = detection.get('spatial_extent_km2', 0) > 0.1 and detection.get('spatial_extent_km2', 0) < 100.0
        
        # Method-specific quality checks
        if method == 'gaussian_plume':
            quality_flags['sufficient_spatial_data'] = len(enhancement_data.get('lats', [])) > 5
        
        return quality_flags
    
    def _calculate_confidence_level(self, quality_flags: Dict[str, bool], met_data: Dict) -> float:
        """Calculate overall confidence level for emission quantification."""
        
        # Weight different quality aspects
        weights = {
            'has_wind_data': 0.2,
            'realistic_wind_speed': 0.15,
            'has_boundary_layer_data': 0.1,
            'significant_enhancement': 0.2,
            'realistic_enhancement': 0.1,
            'high_detection_score': 0.15,
            'reasonable_area': 0.1
        }
        
        # Calculate weighted score
        score = 0.0
        total_weight = 0.0
        
        for flag, is_good in quality_flags.items():
            if flag in weights:
                score += weights[flag] * (1.0 if is_good else 0.0)
                total_weight += weights[flag]
        
        # Normalize to 0-1 range
        confidence = score / total_weight if total_weight > 0 else 0.5
        
        # Apply additional factors
        wind_speed = met_data.get('wind_speed', 0)
        if wind_speed < 2.0:  # Very low wind reduces confidence
            confidence *= 0.7
        elif wind_speed > 15.0:  # Very high wind also reduces confidence
            confidence *= 0.8
        
        return np.clip(confidence, 0.0, 1.0)
    
    def aggregate_emissions(self, results: List[EmissionQuantificationResult],
                          aggregation_method: str = 'sum') -> Dict:
        """Aggregate emission results across multiple detections."""
        
        if not results:
            return {'total_emission_rate_kg_hr': 0, 'count': 0}
        
        emission_rates = [r.emission_rate_kg_hr for r in results]
        uncertainties = [r.emission_rate_uncertainty_kg_hr for r in results]
        confidence_levels = [r.confidence_level for r in results]
        
        if aggregation_method == 'sum':
            total_emission_rate = np.sum(emission_rates)
            total_uncertainty = np.sqrt(np.sum(np.array(uncertainties)**2))  # Error propagation
            
        elif aggregation_method == 'weighted_mean':
            weights = np.array(confidence_levels)
            total_emission_rate = np.average(emission_rates, weights=weights) * len(results)
            total_uncertainty = np.sqrt(np.sum((np.array(uncertainties) * weights)**2)) / np.sum(weights) * len(results)
            
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        return {
            'total_emission_rate_kg_hr': total_emission_rate,
            'total_uncertainty_kg_hr': total_uncertainty,
            'total_emission_rate_kg_day': total_emission_rate * 24,
            'total_emission_rate_tonnes_year': total_emission_rate * 24 * 365.25 / 1000,
            'mean_confidence_level': np.mean(confidence_levels),
            'count': len(results),
            'methods_used': list(set(r.method_used for r in results))
        }
    
    def export_results_to_dataframe(self, results: List[EmissionQuantificationResult]) -> pd.DataFrame:
        """Export quantification results to pandas DataFrame."""
        
        if not results:
            return pd.DataFrame()
        
        records = []
        for result in results:
            record = {
                'emitter_id': result.emitter_id,
                'emission_rate_kg_hr': result.emission_rate_kg_hr,
                'emission_uncertainty_kg_hr': result.emission_rate_uncertainty_kg_hr,
                'emission_rate_kg_day': result.emission_rate_kg_day,
                'emission_rate_tonnes_year': result.emission_rate_tonnes_year,
                'flux_kg_m2_s': result.flux_kg_m2_s,
                'method': result.method_used,
                'confidence_level': result.confidence_level,
                'wind_speed': result.meteorological_data.get('wind_speed'),
                'boundary_layer_height': result.meteorological_data.get('boundary_layer_height'),
                'temperature': result.meteorological_data.get('temperature'),
                'area_km2': result.spatial_data.get('area_km2')
            }
            
            # Add quality flags as columns
            for flag, value in result.quality_flags.items():
                record[f'quality_{flag}'] = value
            
            records.append(record)
        
        return pd.DataFrame(records)