# ============================================================================
# FILE: src/data/preprocessor.py
# ============================================================================
import numpy as np
import xarray as xr
import logging
from typing import Dict, Optional
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class TROPOMIPreprocessor:
    """
    Preprocess TROPOMI methane data for super-emitter detection.
    
    Features:
    - Quality filtering and outlier removal
    - Background calculation and enhancement computation
    - Data smoothing and gap filling
    - Integration with meteorological data
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
    def preprocess_tropomi_data(self, tropomi_data: xr.Dataset, 
                               met_data: Optional[xr.Dataset] = None) -> xr.Dataset:
        """
        Complete preprocessing pipeline for TROPOMI data.
        
        Args:
            tropomi_data: Raw TROPOMI dataset
            met_data: Optional meteorological dataset
            
        Returns:
            Preprocessed dataset ready for super-emitter detection
        """
        
        logger.info("Starting TROPOMI data preprocessing")
        
        # Step 1: Quality filtering
        processed_data = self._apply_quality_filters(tropomi_data)
        
        # Step 2: Remove outliers
        #processed_data = self._remove_outliers(processed_data)
        
        # Step 3: Calculate background and enhancement
        processed_data = self._calculate_background_and_enhancement(processed_data)
        
        # Step 4: Apply smoothing
        processed_data = self._apply_smoothing(processed_data)
        
        # Step 5: Integrate meteorological data
        if met_data is not None:
            processed_data = self._integrate_meteorological_data(processed_data, met_data)
        
        # Step 6: Add derived variables
        processed_data = self._add_derived_variables(processed_data)
        
        logger.info("TROPOMI preprocessing completed")
        return processed_data
    
    def _apply_quality_filters(self, data: xr.Dataset) -> xr.Dataset:
        """Apply quality filters to TROPOMI data."""
        
        logger.info("Applying quality filters")
        
        # Create quality mask
        quality_mask = xr.ones_like(data.ch4, dtype=bool)
        
        # Filter based on QA values
        if 'qa_value' in data.data_vars:
            qa_threshold = self.config.get('data', {}).get('tropomi', {}).get('quality_threshold', 0.5)
            quality_mask = quality_mask & (data.qa_value >= qa_threshold)
        
        # Filter unrealistic values
        quality_mask = quality_mask & (data.ch4 > 1600) & (data.ch4 < 2500)  # Typical atmospheric range
        
        # Apply mask
        filtered_data = data.where(quality_mask)
        
        # Count valid pixels
        valid_pixels = quality_mask.sum()
        total_pixels = quality_mask.size
        coverage = float(valid_pixels) / total_pixels if total_pixels > 0 else 0
        
        logger.info(f"Quality filtering: {coverage:.1%} pixels retained")
        
        return filtered_data
    
    def _remove_outliers(self, data: xr.Dataset) -> xr.Dataset:
        """Remove statistical outliers."""
        
        logger.info("Removing outliers")
        
        ch4_data = data.ch4
        
        # Global outlier removal using IQR method
        q1 = ch4_data.quantile(0.25, dim=['lat', 'lon'])
        q3 = ch4_data.quantile(0.75, dim=['lat', 'lon'])
        iqr = q3 - q1
        
        lower_bound = q1 - 3.5 * iqr
        upper_bound = q3 + 3.5 * iqr
        
        # Create outlier mask
        outlier_mask = (ch4_data >= lower_bound) & (ch4_data <= upper_bound)
        
        # Apply mask
        cleaned_data = data.where(outlier_mask)
        
        # Log statistics
        outliers_removed = (~outlier_mask).sum()
        total_valid = outlier_mask.sum()
        outlier_fraction = float(outliers_removed) / (outliers_removed + total_valid) if (outliers_removed + total_valid) > 0 else 0
        
        logger.info(f"Outlier removal: {outlier_fraction:.2%} of valid pixels removed")
        
        return cleaned_data
    
    def _calculate_background_and_enhancement(self, data: xr.Dataset) -> xr.Dataset:
        """Calculate background concentrations and methane enhancement."""
        
        logger.info("Calculating background and enhancement")
        
        ch4_data = data.ch4
        
        # Method 1: Temporal background (median over time for each pixel)
        temporal_background = ch4_data.median(dim='time')
        
        # Method 2: Spatial background (percentile approach)
        spatial_background = ch4_data.quantile(0.2, dim=['lat', 'lon'])
        
        # Method 3: Local background (spatial median filter)
        local_background = xr.apply_ufunc(
            lambda x: ndimage.median_filter(x, size=5, mode='nearest'),
            ch4_data,
            input_core_dims=[['lat', 'lon']],
            output_core_dims=[['lat', 'lon']],
            dask='parallelized'
        )
        
        # Choose primary background method (temporal for now)
        background = temporal_background
        
        # Calculate enhancement
        enhancement = ch4_data - background
        
        # Add to dataset
        result_data = data.copy()
        result_data['background'] = background
        result_data['temporal_background'] = temporal_background
        result_data['spatial_background'] = spatial_background
        result_data['local_background'] = local_background
        result_data['enhancement'] = enhancement
        
        # Add attributes
        result_data['background'].attrs = {
            'long_name': 'Background CH4 concentration',
            'units': 'ppb',
            'method': 'temporal_median'
        }
        
        result_data['enhancement'].attrs = {
            'long_name': 'CH4 enhancement above background',
            'units': 'ppb',
            'description': 'CH4 concentration minus background'
        }
        
        logger.info("Background and enhancement calculation completed")
        return result_data
    
    def _apply_smoothing(self, data: xr.Dataset, sigma: float = 1.0) -> xr.Dataset:
        """Apply spatial smoothing to reduce noise."""
        
        logger.info(f"Applying spatial smoothing (sigma={sigma})")
        
        smoothed_data = data.copy()
        
        # Apply Gaussian smoothing to enhancement field
        if 'enhancement' in data.data_vars:
            enhancement_smooth = xr.apply_ufunc(
                lambda x: ndimage.gaussian_filter(x, sigma=sigma, mode='nearest'),
                data.enhancement,
                input_core_dims=[['lat', 'lon']],
                output_core_dims=[['lat', 'lon']],
                dask='parallelized'
            )
            
            smoothed_data['enhancement_smooth'] = enhancement_smooth
            smoothed_data['enhancement_smooth'].attrs = {
                'long_name': 'Smoothed CH4 enhancement',
                'units': 'ppb',
                'smoothing_sigma': sigma
            }
        
        # Also smooth the main CH4 field
        ch4_smooth = xr.apply_ufunc(
            lambda x: ndimage.gaussian_filter(x, sigma=sigma, mode='nearest'),
            data.ch4,
            input_core_dims=[['lat', 'lon']],
            output_core_dims=[['lat', 'lon']],
            dask='parallelized'
        )
        
        smoothed_data['ch4_smooth'] = ch4_smooth
        smoothed_data['ch4_smooth'].attrs = {
            'long_name': 'Smoothed CH4 concentration',
            'units': 'ppb',
            'smoothing_sigma': sigma
        }
        
        return smoothed_data
    
    def _integrate_meteorological_data(self, tropomi_data: xr.Dataset, 
                                     met_data: xr.Dataset) -> xr.Dataset:
        """Integrate meteorological data with TROPOMI data."""
        
        logger.info("Integrating meteorological data")
        
        # Interpolate meteorological data to TROPOMI grid
        met_interp = met_data.interp(
            lat=tropomi_data.lat,
            lon=tropomi_data.lon,
            time=tropomi_data.time,
            method='linear'
        )
        
        # Merge datasets
        integrated_data = xr.merge([tropomi_data, met_interp])
        
        logger.info("Meteorological data integration completed")
        return integrated_data
    
    def _add_derived_variables(self, data: xr.Dataset) -> xr.Dataset:
        """Add derived variables useful for super-emitter detection."""
        
        logger.info("Adding derived variables")
        
        result_data = data.copy()
        
        # Enhanced mask (enhancement above threshold)
        if 'enhancement' in data.data_vars:
            enhancement_threshold = 20.0  # ppb
            enhanced_mask = data.enhancement > enhancement_threshold
            result_data['enhanced_mask'] = enhanced_mask
            result_data['enhanced_mask'].attrs = {
                'long_name': 'Enhanced methane pixels',
                'description': f'Pixels with enhancement > {enhancement_threshold} ppb'
            }
        
        # Calculate spatial gradients
        if 'ch4' in data.data_vars:
            # Gradient in latitude direction
            ch4_grad_lat = data.ch4.differentiate('lat')
            result_data['ch4_grad_lat'] = ch4_grad_lat
            result_data['ch4_grad_lat'].attrs = {
                'long_name': 'CH4 gradient in latitude direction',
                'units': 'ppb/degree'
            }
            
            # Gradient in longitude direction
            ch4_grad_lon = data.ch4.differentiate('lon')
            result_data['ch4_grad_lon'] = ch4_grad_lon
            result_data['ch4_grad_lon'].attrs = {
                'long_name': 'CH4 gradient in longitude direction',
                'units': 'ppb/degree'
            }
            
            # Gradient magnitude
            grad_magnitude = np.sqrt(ch4_grad_lat**2 + ch4_grad_lon**2)
            result_data['ch4_grad_magnitude'] = grad_magnitude
            result_data['ch4_grad_magnitude'].attrs = {
                'long_name': 'CH4 gradient magnitude',
                'units': 'ppb/degree'
            }
        
        # Wind-corrected enhancement (if wind data available)
        if 'wind_speed' in data.data_vars and 'enhancement' in data.data_vars:
            # Simple wind correction (normalize by wind speed)
            wind_corrected_enhancement = data.enhancement / (data.wind_speed + 0.1)  # Add small value to avoid division by zero
            result_data['wind_corrected_enhancement'] = wind_corrected_enhancement
            result_data['wind_corrected_enhancement'].attrs = {
                'long_name': 'Wind-corrected CH4 enhancement',
                'units': 'ppb/(m/s)',
                'description': 'Enhancement normalized by wind speed'
            }
        
        # Detection score (simple combination of enhancement and gradient)
        if 'enhancement' in data.data_vars and 'ch4_grad_magnitude' in result_data.data_vars:
            # Normalize both fields
            enh_norm = (data.enhancement - data.enhancement.mean()) / data.enhancement.std()
            grad_norm = (result_data.ch4_grad_magnitude - result_data.ch4_grad_magnitude.mean()) / result_data.ch4_grad_magnitude.std()
            
            # Combine (enhancement is more important than gradient)
            detection_score = 0.7 * enh_norm + 0.3 * grad_norm
            result_data['detection_score'] = detection_score
            result_data['detection_score'].attrs = {
                'long_name': 'Detection score',
                'description': 'Combined score for super-emitter detection'
            }
        
        # Emission potential index
        if 'enhancement' in data.data_vars and 'boundary_layer_height' in data.data_vars:
            # Simple emission potential based on enhancement and mixing
            emission_potential = data.enhancement * data.boundary_layer_height / 1000.0  # Normalize by 1 km
            result_data['emission_potential'] = emission_potential
            result_data['emission_potential'].attrs = {
                'long_name': 'Emission potential index',
                'description': 'Enhancement weighted by boundary layer height'
            }
        
        logger.info("Derived variables added")
        return result_data
    
    def calculate_statistics(self, data: xr.Dataset) -> Dict:
        """Calculate comprehensive statistics for the processed dataset."""
        
        logger.info("Calculating dataset statistics")
        
        stats = {}
        
        # CH4 statistics
        if 'ch4' in data.data_vars:
            ch4_data = data.ch4.values
            valid_ch4 = ch4_data[~np.isnan(ch4_data)]
            
            if len(valid_ch4) > 0:
                stats['ch4'] = {
                    'mean': float(np.mean(valid_ch4)),
                    'std': float(np.std(valid_ch4)),
                    'min': float(np.min(valid_ch4)),
                    'max': float(np.max(valid_ch4)),
                    'median': float(np.median(valid_ch4)),
                    'count': len(valid_ch4),
                    'coverage': len(valid_ch4) / len(ch4_data.flatten())
                }
        
        # Enhancement statistics
        if 'enhancement' in data.data_vars:
            enh_data = data.enhancement.values
            valid_enh = enh_data[~np.isnan(enh_data)]
            
            if len(valid_enh) > 0:
                stats['enhancement'] = {
                    'mean': float(np.mean(valid_enh)),
                    'std': float(np.std(valid_enh)),
                    'min': float(np.min(valid_enh)),
                    'max': float(np.max(valid_enh)),
                    'median': float(np.median(valid_enh)),
                    'count': len(valid_enh),
                    'positive_enhancement_fraction': float(np.sum(valid_enh > 0) / len(valid_enh))
                }
        
        # Enhanced pixels statistics
        if 'enhanced_mask' in data.data_vars:
            enhanced_pixels = data.enhanced_mask.sum()
            total_pixels = data.enhanced_mask.count()
            
            stats['enhanced_pixels'] = {
                'count': int(enhanced_pixels),
                'total_pixels': int(total_pixels),
                'fraction': float(enhanced_pixels / total_pixels) if total_pixels > 0 else 0
            }
        
        # Temporal coverage
        stats['temporal'] = {
            'start_time': str(data.time.min().values),
            'end_time': str(data.time.max().values),
            'time_steps': len(data.time),
            'time_span_days': float((data.time.max() - data.time.min()) / np.timedelta64(1, 'D'))
        }
        
        # Spatial coverage
        stats['spatial'] = {
            'lat_range': [float(data.lat.min()), float(data.lat.max())],
            'lon_range': [float(data.lon.min()), float(data.lon.max())],
            'lat_resolution': float(data.lat.diff('lat').mean()) if len(data.lat) > 1 else 0,
            'lon_resolution': float(data.lon.diff('lon').mean()) if len(data.lon) > 1 else 0,
            'grid_cells': len(data.lat) * len(data.lon)
        }
        
        return stats
    
    def validate_processed_data(self, data: xr.Dataset) -> Dict:
        """Validate the processed dataset for quality issues."""
        
        logger.info("Validating processed data")
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'quality_score': 1.0
        }
        
        # Check for required variables
        required_vars = ['ch4', 'background', 'enhancement']
        for var in required_vars:
            if var not in data.data_vars:
                validation_results['errors'].append(f"Missing required variable: {var}")
                validation_results['is_valid'] = False
        
        # Check data coverage
        if 'ch4' in data.data_vars:
            ch4_coverage = (~np.isnan(data.ch4)).sum() / data.ch4.size
            if ch4_coverage < 0.1:
                validation_results['errors'].append(f"Very low data coverage: {ch4_coverage:.1%}")
                validation_results['is_valid'] = False
            elif ch4_coverage < 0.5:
                validation_results['warnings'].append(f"Low data coverage: {ch4_coverage:.1%}")
                validation_results['quality_score'] *= 0.8
        
        # Check for unrealistic values
        if 'enhancement' in data.data_vars:
            max_enhancement = float(data.enhancement.max())
            if max_enhancement > 500:  # Very high enhancement
                validation_results['warnings'].append(f"Very high enhancement detected: {max_enhancement:.1f} ppb")
            
            # Check for too many negative enhancements
            negative_fraction = float((data.enhancement < -50).sum() / data.enhancement.count())
            if negative_fraction > 0.3:
                validation_results['warnings'].append(f"High fraction of negative enhancements: {negative_fraction:.1%}")
                validation_results['quality_score'] *= 0.9
        
        # Check temporal consistency
        if len(data.time) < 2:
            validation_results['warnings'].append("Only single time step available")
        
        # Final quality score
        if validation_results['errors']:
            validation_results['quality_score'] = 0.0
        elif validation_results['warnings']:
            validation_results['quality_score'] *= max(0.5, 1.0 - 0.1 * len(validation_results['warnings']))
        
        logger.info(f"Data validation completed. Quality score: {validation_results['quality_score']:.2f}")
        
        return validation_results