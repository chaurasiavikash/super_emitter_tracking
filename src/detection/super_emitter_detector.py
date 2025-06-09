# ============================================================================
# FILE: src/detection/super_emitter_detector.py (FIXED VERSION)
# ============================================================================
import numpy as np
import pandas as pd
import xarray as xr
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from scipy import ndimage
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings

# Handle geopandas import gracefully
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("⚠️ GeoPandas not available - facility association will be disabled")

logger = logging.getLogger(__name__)

class SuperEmitterDetector:
    """
    Advanced super-emitter detection system for TROPOMI methane data.
    
    Features:
    - Multi-algorithm detection with ensemble voting
    - Temporal persistence filtering
    - Statistical significance testing
    - Integration with known facility databases
    - Uncertainty quantification
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.detection_params = config['super_emitters']['detection']
        # Handle both config structures
        # Handle both config structures
        if 'super_emitters' in config and 'background' in config['super_emitters']:
            self.background_params = config['super_emitters']['background']
        else:
            self.background_params = config['background']  # Direct access for YAML config

        # Ensure required keys exist with defaults
        self.background_params.setdefault('spatial_radius_km', 100)
        self.background_params.setdefault('percentile', 20)
        self.background_params.setdefault('window_days', 30)
        self.background_params.setdefault('method', 'local_median')
        self.clustering_params = config['super_emitters']['clustering']
        
        # Initialize detection history
        self.detection_history = []
        self.known_emitters = None
        self.background_cache = {}
        
        # Load known facility database if available
        self._load_facility_database()
        
        logger.info("SuperEmitterDetector initialized")
        
    def _load_facility_database(self):
        """Load known facility database for validation and tracking."""
        if not GEOPANDAS_AVAILABLE:
            logger.warning("GeoPandas not available - creating empty facility database")
            self.known_facilities = self._create_empty_facility_dataframe()
            return

        try:
            # Simple relative path that works from anywhere in the project
            import os
            facility_file = os.path.join(os.getcwd(), "data", "super_emitters", "facility_database.geojson")

            # If not found, try going up one level
            if not os.path.exists(facility_file):
                facility_file = os.path.join(os.path.dirname(os.getcwd()), "data", "super_emitters", "facility_database.geojson")

            self.known_facilities = gpd.read_file(facility_file)
            logger.info(f"Loaded {len(self.known_facilities)} known facilities from {facility_file}")
        except FileNotFoundError:
            logger.warning(f"No facility database found. Creating empty database.")
            self.known_facilities = self._create_empty_facility_dataframe()
        except Exception as e:
            logger.error(f"Error loading facility database: {e}")
            self.known_facilities = self._create_empty_facility_dataframe()
    
    def _create_empty_facility_dataframe(self):
        """Create empty facility dataframe structure."""
        if GEOPANDAS_AVAILABLE:
            return gpd.GeoDataFrame(
                columns=['facility_id', 'name', 'type', 'capacity', 'geometry']
            )
        else:
            return pd.DataFrame(
                columns=['facility_id', 'name', 'type', 'capacity', 'lat', 'lon']
            )
    
    def detect_super_emitters(self, ds: xr.Dataset, 
                            validation_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Main super-emitter detection pipeline.
        
        Args:
            ds: TROPOMI methane dataset
            validation_data: Optional ground truth data for validation
            
        Returns:
            Dictionary containing detection results and metadata
        """
        logger.info("Starting super-emitter detection pipeline")
        
        # Step 1: Preprocess and calculate background
        logger.info("Calculating background concentrations")
        ds_with_bg = self._calculate_background(ds)
        
        # Step 2: Statistical detection
        logger.info("Performing statistical detection")
        statistical_detections = self._statistical_detection(ds_with_bg)
        
        # Step 3: Spatial clustering
        logger.info("Clustering spatial detections")
        clustered_detections = self._spatial_clustering(statistical_detections)
        
        # Step 4: Temporal persistence filtering
        logger.info("Applying temporal persistence filter")
        persistent_detections = self._temporal_persistence_filter(clustered_detections)
        
        # Step 5: Super-emitter classification
        logger.info("Classifying super-emitters")
        super_emitters = self._classify_super_emitters(persistent_detections, ds_with_bg)
        
        # Step 6: Facility association
        logger.info("Associating with known facilities")
        super_emitters = self._associate_with_facilities(super_emitters)
        
        # Step 7: Uncertainty quantification
        logger.info("Calculating uncertainties")
        super_emitters = self._calculate_uncertainties(super_emitters, ds_with_bg)
        
        # Step 8: Validation if data available
        if validation_data is not None:
            logger.info("Validating against ground truth")
            validation_results = self._validate_detections(super_emitters, validation_data)
        else:
            validation_results = None
        
        # Compile results
        results = {
            'super_emitters': super_emitters,
            'detection_metadata': {
                'total_detections': len(super_emitters),
                'detection_timestamp': datetime.now(),
                'data_period': {
                    'start': str(ds.time.min().values),
                    'end': str(ds.time.max().values)
                },
                'parameters_used': self.detection_params.copy()
            },
            'validation_results': validation_results,
            'quality_flags': self._generate_quality_flags(super_emitters, ds_with_bg)
        }
        
        logger.info(f"Detection complete. Found {len(super_emitters)} super-emitters")
        return results
    
    # def _calculate_background(self, ds: xr.Dataset) -> xr.Dataset:
    #     """Calculate background methane concentrations using multiple methods."""
        
    #     method = self.background_params['method']
        
    #     # TEMPORARY FIX: Always use local_median to avoid rolling percentile bug
    #     if method == "rolling_percentile":
    #         logger.info("Using local_median instead of rolling_percentile to avoid bug")
    #         background = self._local_median_background(ds)
    #     elif method == "seasonal":
    #         background = self._seasonal_background(ds)
    #     elif method == "local_median":
    #         #background = self._local_median_background(ds)
    #         background=ds.ch4.median(dim='time')  # Simple temporal median
    #     else:
    #         raise ValueError(f"Unknown background method: {method}")
        
    #     # Calculate enhancement
    #     enhancement = ds.ch4 - background
        
    #     # Store in dataset
    #     ds_result = ds.copy()
    #     ds_result['background'] = background
    #     ds_result['enhancement'] = enhancement
        
    #     return ds_result
    def _calculate_background(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate background methane concentrations using simple approach."""

        # Simple approach: use 20th percentile of all valid data as background
        background_value = ds.ch4.quantile(0.2, skipna=True)

        # Create background field (constant value everywhere)
        background = xr.full_like(ds.ch4, background_value)

        # Calculate enhancement
        enhancement = ds.ch4 - background

        # Store in dataset
        ds_result = ds.copy()
        ds_result['background'] = background
        ds_result['enhancement'] = enhancement

        return ds_result
        
    def _rolling_percentile_background(self, ds: xr.Dataset) -> xr.DataArray:
        """Calculate rolling percentile background - FIXED VERSION."""

        percentile = self.background_params['percentile']
        window_days = self.background_params['window_days']

        # Check if we have enough time steps
        if len(ds.time) < window_days:
            logger.warning(f"Not enough time steps ({len(ds.time)}) for rolling window ({window_days}). Using global percentile.")
            return ds.ch4.quantile(percentile / 100.0, dim='time')

        # FIXED: Simple rolling quantile without axis parameter
        def rolling_quantile(data, q):
            """Apply quantile to rolling windows."""
            return np.nanquantile(data, q)  # Remove axis=0

        try:
            # Calculate rolling percentile over time dimension
            background = ds.ch4.rolling(
                time=window_days, center=True, min_periods=max(1, window_days//2)
            ).reduce(rolling_quantile, q=percentile/100.0)

            # Fill NaN values at edges with global percentile
            global_percentile = ds.ch4.quantile(percentile / 100.0, dim='time')
            background = background.fillna(global_percentile)

        except Exception as e:
            logger.warning(f"Rolling percentile calculation failed: {e}. Using global percentile.")
            background = ds.ch4.quantile(percentile / 100.0, dim='time')

        return background
    
    def _seasonal_background(self, ds: xr.Dataset) -> xr.DataArray:
        """Calculate seasonal background using harmonic decomposition."""
        
        time_numeric = (ds.time - ds.time[0]) / np.timedelta64(1, 'D')
        
        # Fit seasonal model for each pixel
        def fit_seasonal(ts):
            if np.isnan(ts).all():
                return ts
            
            valid_mask = ~np.isnan(ts)
            if valid_mask.sum() < 10:  # Need minimum data points
                return np.full_like(ts, np.nanmedian(ts))
            
            t = time_numeric.values[valid_mask]
            y = ts[valid_mask]
            
            # Simple harmonic model: a + b*cos(2π*t/365) + c*sin(2π*t/365)
            X = np.column_stack([
                np.ones(len(t)),
                np.cos(2 * np.pi * t / 365.25),
                np.sin(2 * np.pi * t / 365.25)
            ])
            
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                
                # Predict for all times
                X_full = np.column_stack([
                    np.ones(len(time_numeric)),
                    np.cos(2 * np.pi * time_numeric / 365.25),
                    np.sin(2 * np.pi * time_numeric / 365.25)
                ])
                
                return X_full @ coeffs
                
            except np.linalg.LinAlgError:
                return np.full_like(ts, np.nanmedian(ts))
        
        # Apply to each pixel
        background = xr.apply_ufunc(
            fit_seasonal,
            ds.ch4,
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            dask='parallelized',
            output_dtypes=[float]
        )
        
        return background
    
    def _local_median_background(self, ds: xr.Dataset) -> xr.DataArray:
        """Calculate local median background."""
        
        radius_km = self.background_params['spatial_radius_km']
        
        # Convert radius to approximate pixels (assuming ~7km TROPOMI resolution)
        radius_pixels = max(1, int(radius_km / 7))
        
        # Calculate spatial median for each time step
        def spatial_median(data_2d):
            return ndimage.median_filter(data_2d, size=radius_pixels, mode='nearest')
        
        background = xr.apply_ufunc(
            spatial_median,
            ds.ch4,
            input_core_dims=[['lat', 'lon']],
            output_core_dims=[['lat', 'lon']],
            dask='parallelized'
            ).fillna(ds.ch4.mean())  # Fill NaN with mean value
        
        return background
    
    def _statistical_detection(self, ds: xr.Dataset) -> xr.Dataset:
        """Perform statistical detection of anomalies."""
        
        enhancement = ds.enhancement
        threshold = self.detection_params['enhancement_threshold']
        
        # Method 1: Simple threshold
        threshold_mask = enhancement > threshold
        
        # Method 2: Statistical outlier detection
        # Method 2: Statistical outlier detection - FIXED
        enhancement_std = enhancement.std(dim=['lat', 'lon'], skipna=True)
        enhancement_mean = enhancement.mean(dim=['lat', 'lon'], skipna=True)
        # Avoid division by zero/NaN
        enhancement_std = enhancement_std.where(enhancement_std > 0, 1.0)
        z_scores = (enhancement - enhancement_mean) / enhancement_std
        z_scores = z_scores.fillna(0)  # Replace NaN with 0
        statistical_mask = z_scores > 1.0  # Lower threshold
        
        # Method 3: Local outlier detection
        local_mask = self._local_outlier_detection(enhancement)
        
        # Combine methods with voting
        detection_score = (
            threshold_mask.astype(float) + 
            statistical_mask.astype(float) + 
            local_mask.astype(float)
        ) / 3.0
        
        # Apply confidence threshold
        # Apply confidence threshold
        confidence_threshold = self.detection_params['confidence_threshold']
        final_mask = detection_score >= confidence_threshold

        # DEBUG PRINTS (keep only these):
        # print(f"🔍 DEBUG: Enhancement threshold: {threshold}")
        # print(f"🔍 DEBUG: Max enhancement: {enhancement.max().values:.1f}")
        # print(f"🔍 DEBUG: Pixels above threshold: {threshold_mask.sum().values}")
        # print(f"🔍 DEBUG: Detection pixels found: {final_mask.sum().values}")
        # print(f"🔍 DEBUG: Max detection score: {detection_score.max().values:.3f}")
        # print(f"🔍 DEBUG: Confidence threshold: {confidence_threshold}")

        # Store results
        ds_result = ds.copy()
        ds_result['detection_mask'] = final_mask
        ds_result['detection_score'] = detection_score
        ds_result['z_scores'] = z_scores

        return ds_result
    
    def _local_outlier_detection(self, enhancement: xr.DataArray) -> xr.DataArray:
        """Detect local outliers using spatial statistics."""
        
        def detect_outliers_2d(data_2d):
            if np.isnan(data_2d).all():
                return np.zeros_like(data_2d, dtype=bool)
            
            # Calculate local statistics
            local_mean = ndimage.uniform_filter(data_2d, size=5, mode='nearest')
            local_std = ndimage.generic_filter(
                data_2d, np.nanstd, size=5, mode='nearest'
            )
            
            # Avoid division by zero
            local_std = np.maximum(local_std, 0.1)
            
            # Calculate local z-scores
            local_z = (data_2d - local_mean) / local_std
            
            return local_z > 2.5
        
        outlier_mask = xr.apply_ufunc(
            detect_outliers_2d,
            enhancement,
            input_core_dims=[['lat', 'lon']],
            output_core_dims=[['lat', 'lon']],
            dask='parallelized'
        )
        
        return outlier_mask
    
    def _spatial_clustering(self, ds: xr.Dataset) -> List[Dict]:
        """Cluster spatially connected detections."""
        
        detections = []
        
        for t, time_val in enumerate(ds.time.values):
            detection_2d = ds.detection_mask.isel(time=t).values
            enhancement_2d = ds.enhancement.isel(time=t).values
            score_2d = ds.detection_score.isel(time=t).values
            
            if not np.any(detection_2d):
                continue
            
            # Get coordinates of detected pixels
            lat_indices, lon_indices = np.where(detection_2d)
            
            # if len(lat_indices) < self.clustering_params['min_samples']:
            #     continue
            if len(lat_indices) < 1:  # Accept any detections
                continue
            
            # Convert to geographic coordinates
            lats = ds.lat.values[lat_indices]
            lons = ds.lon.values[lon_indices]
            enhancements = enhancement_2d[lat_indices, lon_indices]
            scores = score_2d[lat_indices, lon_indices]
            
            # Convert to Cartesian coordinates for clustering
            coords = self._geo_to_cartesian(lats, lons)
            
            # Clustering
            if self.clustering_params['algorithm'] == 'DBSCAN':
                clusters = self._dbscan_clustering(coords, lats, lons, enhancements, scores)
            else:
                clusters = self._connected_components_clustering(
                    lat_indices, lon_indices, lats, lons, enhancements, scores
                )
            
            # Add timestamp to each cluster
            for cluster in clusters:
                cluster['timestamp'] = time_val
                cluster['time_index'] = t
                
            detections.extend(clusters)
            
            #print(f"🔍 DEBUG: Time step {t}: found {len(clusters)} clusters")
        
        logger.info(f"Found {len(detections)} spatial clusters")
        return detections
    
    def _geo_to_cartesian(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Convert geographic coordinates to Cartesian for distance calculations."""
        
        # Simple approximation for small regions
        lat_center = np.mean(lats)
        lon_center = np.mean(lons)
        
        # Convert to km using approximate conversion
        lat_km = (lats - lat_center) * 111.32  # 1 degree ≈ 111.32 km
        lon_km = (lons - lon_center) * 111.32 * np.cos(np.radians(lat_center))
        
        return np.column_stack([lat_km, lon_km])
    
    def _dbscan_clustering(self, coords: np.ndarray, lats: np.ndarray, 
                          lons: np.ndarray, enhancements: np.ndarray, 
                          scores: np.ndarray) -> List[Dict]:
        """Perform DBSCAN clustering."""
        
        eps_km = self.clustering_params['eps_km']
        min_samples = self.clustering_params['min_samples']
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps_km, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(coords)
        
        clusters = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            cluster_mask = cluster_labels == label
            cluster_lats = lats[cluster_mask]
            cluster_lons = lons[cluster_mask]
            cluster_enhancements = enhancements[cluster_mask]
            cluster_scores = scores[cluster_mask]
            
            # Calculate cluster properties
            cluster = {
                'cluster_id': len(clusters),
                'n_pixels': len(cluster_lats),
                'center_lat': float(np.mean(cluster_lats)),
                'center_lon': float(np.mean(cluster_lons)),
                'lat_extent': float(np.max(cluster_lats) - np.min(cluster_lats)),
                'lon_extent': float(np.max(cluster_lons) - np.min(cluster_lons)),
                'max_enhancement': float(np.max(cluster_enhancements)),
                'mean_enhancement': float(np.mean(cluster_enhancements)),
                'total_enhancement': float(np.sum(cluster_enhancements)),
                'mean_score': float(np.mean(cluster_scores)),
                'lats': cluster_lats.tolist(),
                'lons': cluster_lons.tolist(),
                'enhancements': cluster_enhancements.tolist()
            }
            
            clusters.append(cluster)
        
        return clusters
    
    def _connected_components_clustering(self, lat_indices: np.ndarray, 
                                       lon_indices: np.ndarray, lats: np.ndarray,
                                       lons: np.ndarray, enhancements: np.ndarray,
                                       scores: np.ndarray) -> List[Dict]:
        """Alternative clustering using connected components."""
        
        # Create binary mask
        # Create binary mask - fix indexing
        mask_shape = (np.max(lat_indices) - np.min(lat_indices) + 1, 
              np.max(lon_indices) - np.min(lon_indices) + 1)
        binary_mask = np.zeros(mask_shape, dtype=bool)
        
        # Map indices to mask coordinates
        lat_min, lon_min = np.min(lat_indices), np.min(lon_indices)
        mask_lat_indices = lat_indices - lat_min
        mask_lon_indices = lon_indices - lon_min
        
        binary_mask[mask_lat_indices, mask_lon_indices] = True
        
        # Find connected components
        labeled_array, num_features = ndimage.label(binary_mask)
        
        clusters = []
        for cluster_id in range(1, num_features + 1):
            cluster_coords = np.where(labeled_array == cluster_id)
            
            # Map back to original coordinates
            orig_lat_indices = cluster_coords[0] + lat_min
            orig_lon_indices = cluster_coords[1] + lon_min
            
            # Get data for this cluster
            data_indices = []
            for i, (lat_idx, lon_idx) in enumerate(zip(lat_indices, lon_indices)):
                if lat_idx in orig_lat_indices and lon_idx in orig_lon_indices:
                    data_indices.append(i)
            
            # if len(data_indices) < self.clustering_params['min_samples']:
            #     continue
            if len(data_indices) < 1:  # Accept any detections
                continue
            cluster_lats = lats[data_indices]
            cluster_lons = lons[data_indices]
            cluster_enhancements = enhancements[data_indices]
            cluster_scores = scores[data_indices]
            
            cluster = {
                'cluster_id': len(clusters),
                'n_pixels': len(cluster_lats),
                'center_lat': float(np.mean(cluster_lats)),
                'center_lon': float(np.mean(cluster_lons)),
                'lat_extent': float(np.max(cluster_lats) - np.min(cluster_lats)),
                'lon_extent': float(np.max(cluster_lons) - np.min(cluster_lons)),
                'max_enhancement': float(np.max(cluster_enhancements)),
                'mean_enhancement': float(np.mean(cluster_enhancements)),
                'total_enhancement': float(np.sum(cluster_enhancements)),
                'mean_score': float(np.mean(cluster_scores)),
                'lats': cluster_lats.tolist(),
                'lons': cluster_lons.tolist(),
                'enhancements': cluster_enhancements.tolist()
            }
            
            clusters.append(cluster)
        
        return clusters
    
    def _temporal_persistence_filter(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections based on temporal persistence."""
        
        min_detections = self.config['tracking']['persistence']['min_detections']
        max_gap_days = self.config['tracking']['persistence']['max_gap_days']
        
        if not detections:
            return []
        
        # Group detections by spatial proximity
        spatial_groups = self._group_by_spatial_proximity(detections)
        
        persistent_detections = []
        
        for group in spatial_groups:
            # Sort by time
            group_sorted = sorted(group, key=lambda x: x['timestamp'])
            
            # Check temporal persistence
            if len(group_sorted) < min_detections:
                continue
            
            # Check for excessive gaps
            timestamps = [pd.Timestamp(d['timestamp']) for d in group_sorted]
            gaps = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
            
            if any(gap > max_gap_days for gap in gaps):
                # Split at large gaps
                sub_groups = self._split_at_gaps(group_sorted, max_gap_days)
                for sub_group in sub_groups:
                    if len(sub_group) >= min_detections:
                        persistent_detections.extend(sub_group)
            else:
                persistent_detections.extend(group_sorted)
        
        logger.info(f"Filtered to {len(persistent_detections)} persistent detections")
        return persistent_detections
    
    def _group_by_spatial_proximity(self, detections: List[Dict]) -> List[List[Dict]]:
        """Group detections by spatial proximity."""
        
        if not detections:
            return []
        
        # Extract coordinates
        coords = np.array([[d['center_lat'], d['center_lon']] for d in detections])
        
        # Convert to Cartesian
        coords_km = self._geo_to_cartesian(coords[:, 0], coords[:, 1])
        
        # Clustering
        eps_km = self.clustering_params['eps_km'] * 2  # Larger radius for persistence
        dbscan = DBSCAN(eps=eps_km, min_samples=1)
        labels = dbscan.fit_predict(coords_km)
        
        # Group by labels
        groups = []
        for label in np.unique(labels):
            if label == -1:
                # Add noise points as individual groups
                noise_indices = np.where(labels == label)[0]
                for idx in noise_indices:
                    groups.append([detections[idx]])
            else:
                group_indices = np.where(labels == label)[0]
                group = [detections[idx] for idx in group_indices]
                groups.append(group)
        
        return groups
    
    def _split_at_gaps(self, detections: List[Dict], max_gap_days: int) -> List[List[Dict]]:
        """Split detection sequence at large temporal gaps."""
        
        if len(detections) <= 1:
            return [detections]
        
        sub_groups = []
        current_group = [detections[0]]
        
        for i in range(1, len(detections)):
            prev_time = pd.Timestamp(detections[i-1]['timestamp'])
            curr_time = pd.Timestamp(detections[i]['timestamp'])
            gap_days = (curr_time - prev_time).days
            
            if gap_days <= max_gap_days:
                current_group.append(detections[i])
            else:
                # Start new group
                if len(current_group) > 0:
                    sub_groups.append(current_group)
                current_group = [detections[i]]
        
        if len(current_group) > 0:
            sub_groups.append(current_group)
        
        return sub_groups
    
    def _classify_super_emitters(self, detections: List[Dict], ds: xr.Dataset) -> pd.DataFrame:
        """Classify detections as super-emitters based on emission criteria."""
        #print(f"🔍 DEBUG: Classifying {len(detections)} detections")
        emission_threshold = self.detection_params['emission_rate_threshold']
        
        super_emitters = []
        
        for detection in detections:
            # Estimate emission rate (simplified approach)
            emission_rate = self._estimate_emission_rate(detection, ds)
            
            # Check if it qualifies as super-emitter
            if emission_rate >= emission_threshold:
                emitter = {
                    'emitter_id': f"SE_{len(super_emitters):04d}",
                    'center_lat': detection['center_lat'],
                    'center_lon': detection['center_lon'],
                    'first_detected': detection['timestamp'],
                    'last_detected': detection['timestamp'],
                    'max_enhancement': detection['max_enhancement'],
                    'mean_enhancement': detection['mean_enhancement'],
                    'estimated_emission_rate_kg_hr': emission_rate,
                    'detection_score': detection['mean_score'],
                    'spatial_extent_km2': self._calculate_area(detection),
                    'n_detections': 1,
                    'facility_id': None,  # Will be filled in association step
                    'facility_name': None,
                    'facility_type': None
                }
                super_emitters.append(emitter)
        
        # Convert to DataFrame
        if super_emitters:
            df = pd.DataFrame(super_emitters)
            # Aggregate multiple detections of same emitter
            df = self._aggregate_multiple_detections(df)
        else:
            df = pd.DataFrame(columns=[
                'emitter_id', 'center_lat', 'center_lon', 'first_detected',
                'last_detected', 'max_enhancement', 'mean_enhancement',
                'estimated_emission_rate_kg_hr', 'detection_score', 
                'spatial_extent_km2', 'n_detections', 'facility_id',
                'facility_name', 'facility_type'
            ])
        
        logger.info(f"Classified {len(df)} super-emitters")
        return df
    
    def _estimate_emission_rate(self, detection: Dict, ds: xr.Dataset) -> float:
        """Estimate emission rate for a detection (simplified approach)."""
        
        # This is a simplified emission estimation
        # In practice, you'd use more sophisticated atmospheric models
        
        enhancement = detection['mean_enhancement']  # ppb
        area_km2 = self._calculate_area(detection)
        
        # Assumptions for quick estimation
        boundary_layer_height = 1000.0  # meters
        wind_speed = 5.0  # m/s (should come from meteorological data)
        mixing_ratio_to_mass = 0.67  # kg/m3 per ppm CH4
        
        # Convert enhancement to mass flux
        # Very simplified: Enhancement * Area * Wind * Conversion
        emission_rate = (
            enhancement * 1e-3 *  # ppb to ppm
            area_km2 * 1e6 *      # km2 to m2
            boundary_layer_height *
            wind_speed / 1000.0 *  # m/s to km/s
            mixing_ratio_to_mass *
            3600.0  # convert to per hour
        )
        
        return max(0.0, emission_rate)
    
    def _calculate_area(self, detection: Dict) -> float:
        """Calculate approximate area of detection in km2."""
        
        # Simple approximation using extent
        lat_extent = detection['lat_extent']
        lon_extent = detection['lon_extent']
        
        # Convert to km (rough approximation)
        lat_km = lat_extent * 111.32
        lon_km = lon_extent * 111.32 * np.cos(np.radians(detection['center_lat']))
        
        return lat_km * lon_km
    
    def _aggregate_multiple_detections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate multiple detections of the same emitter."""
        
        if len(df) <= 1:
            return df
        
        # Group by spatial proximity
        coords = df[['center_lat', 'center_lon']].values
        coords_km = self._geo_to_cartesian(coords[:, 0], coords[:, 1])
        
        # Clustering to find same emitters
        eps_km = self.clustering_params['eps_km']
        dbscan = DBSCAN(eps=eps_km, min_samples=1)
        labels = dbscan.fit_predict(coords_km)
        
        aggregated = []
        
        for label in np.unique(labels):
            group_mask = labels == label
            group_df = df[group_mask]
            
            if len(group_df) == 1:
                aggregated.append(group_df.iloc[0].to_dict())
            else:
                # Aggregate multiple detections
                agg_emitter = {
                    'emitter_id': group_df.iloc[0]['emitter_id'],
                    'center_lat': group_df['center_lat'].mean(),
                    'center_lon': group_df['center_lon'].mean(),
                    'first_detected': group_df['first_detected'].min(),
                    'last_detected': group_df['last_detected'].max(),
                    'max_enhancement': group_df['max_enhancement'].max(),
                    'mean_enhancement': group_df['mean_enhancement'].mean(),
                    'estimated_emission_rate_kg_hr': group_df['estimated_emission_rate_kg_hr'].mean(),
                    'detection_score': group_df['detection_score'].mean(),
                    'spatial_extent_km2': group_df['spatial_extent_km2'].max(),
                    'n_detections': len(group_df),
                    'facility_id': None,
                    'facility_name': None,
                    'facility_type': None
                }
                aggregated.append(agg_emitter)
        
        return pd.DataFrame(aggregated)
    
    def _associate_with_facilities(self, super_emitters: pd.DataFrame) -> pd.DataFrame:
        """Associate super-emitters with known facilities."""
        
        if len(super_emitters) == 0 or len(self.known_facilities) == 0:
            return super_emitters
        
        if not GEOPANDAS_AVAILABLE:
            logger.warning("GeoPandas not available - skipping facility association")
            return super_emitters
        
        buffer_km = self.config['super_emitters']['database']['facility_buffer_km']
        
        # Create GeoDataFrame for super-emitters
        emitter_points = [Point(lon, lat) for lat, lon in 
                         zip(super_emitters['center_lat'], super_emitters['center_lon'])]
        emitters_gdf = gpd.GeoDataFrame(super_emitters, geometry=emitter_points, crs='EPSG:4326')
        
        # Buffer facilities
        # Buffer facilities with proper CRS handling
        # facilities_buffered = self.known_facilities.copy()
        # if len(facilities_buffered) > 0:
        #     # Convert to a projected CRS for accurate buffering
        #     facilities_buffered = facilities_buffered.to_crs('EPSG:3857')  # Web Mercator
        #     facilities_buffered['geometry'] = facilities_buffered.geometry.buffer(buffer_km * 1000)  # Convert km to meters
        #     facilities_buffered = facilities_buffered.to_crs('EPSG:4326')  # Back to lat/lon
        # Buffer facilities
        facilities_buffered = self.known_facilities.copy()
        facilities_buffered['geometry'] = facilities_buffered.geometry.buffer(
            buffer_km / 111.32  # Convert km to degrees (rough)
        )
        # Spatial join
        joined = gpd.sjoin(emitters_gdf, facilities_buffered, how='left', predicate='within')
        
        # Update facility information
        if 'facility_id' in joined.columns:
            super_emitters['facility_id'] = joined['facility_id']
        if 'name' in joined.columns:
            super_emitters['facility_name'] = joined['name']
        if 'type' in joined.columns:
            super_emitters['facility_type'] = joined['type']
        
        n_associated = super_emitters['facility_id'].notna().sum()
        logger.info(f"Associated {n_associated} super-emitters with known facilities")
        
        return super_emitters
    
    def _calculate_uncertainties(self, super_emitters: pd.DataFrame, 
                               ds: xr.Dataset) -> pd.DataFrame:
        """Calculate uncertainties for super-emitter detections."""
        
        if len(super_emitters) == 0:
            return super_emitters
        
        # Simple uncertainty estimation
        # In practice, you'd use more sophisticated error propagation
        
        uncertainties = []
        
        for _, emitter in super_emitters.iterrows():
            # Retrieval uncertainty (typical TROPOMI uncertainty ~1-2%)
            retrieval_uncertainty = 0.02 * emitter['mean_enhancement']
            
            # Background uncertainty (estimated from spatial variability)
            background_uncertainty = 5.0  # ppb
            
            # Atmospheric model uncertainty (for emission rates)
            emission_uncertainty = 0.5 * emitter['estimated_emission_rate_kg_hr']
            
            # Combined uncertainty
            total_enhancement_uncertainty = np.sqrt(
                retrieval_uncertainty**2 + background_uncertainty**2
            )
            
            uncertainties.append({
                'enhancement_uncertainty_ppb': total_enhancement_uncertainty,
                'emission_uncertainty_kg_hr': emission_uncertainty,
                'detection_confidence': emitter['detection_score']
            })
        
        # Add uncertainties to DataFrame
        uncertainty_df = pd.DataFrame(uncertainties)
        for col in uncertainty_df.columns:
            super_emitters[col] = uncertainty_df[col]
        
        return super_emitters
    
    def _validate_detections(self, super_emitters: pd.DataFrame, 
                           validation_data: pd.DataFrame) -> Dict:
        """Validate detections against ground truth data."""
        
        if len(super_emitters) == 0 or len(validation_data) == 0:
            return {'validation_available': False}
        
        if not GEOPANDAS_AVAILABLE:
            logger.warning("GeoPandas not available - skipping validation")
            return {'validation_available': False}
        
        matching_radius = self.config.get('validation', {}).get('ground_truth', {}).get('matching_radius_km', 10.0)
        
        # Convert to GeoDataFrames
        emitter_points = [Point(lon, lat) for lat, lon in 
                         zip(super_emitters['center_lat'], super_emitters['center_lon'])]
        emitters_gdf = gpd.GeoDataFrame(super_emitters, geometry=emitter_points, crs='EPSG:4326')
        
        validation_points = [Point(lon, lat) for lat, lon in 
                           zip(validation_data['lat'], validation_data['lon'])]
        validation_gdf = gpd.GeoDataFrame(validation_data, geometry=validation_points, crs='EPSG:4326')
        
        # Find matches within radius
        matches = []
        for _, emitter in emitters_gdf.iterrows():
            distances = validation_gdf.geometry.distance(emitter.geometry) * 111.32  # Convert to km
            close_validation = validation_gdf[distances <= matching_radius]
            
            if len(close_validation) > 0:
                closest = close_validation.iloc[distances[distances <= matching_radius].argmin()]
                matches.append({
                    'emitter_id': emitter['emitter_id'],
                    'validation_id': closest.name,
                    'distance_km': distances.iloc[closest.name],
                    'emitter_emission': emitter['estimated_emission_rate_kg_hr'],
                    'validation_emission': closest.get('emission_rate_kg_hr', np.nan)
                })
        
        # Calculate validation metrics
        n_detections = len(super_emitters)
        n_validation = len(validation_data)
        n_matches = len(matches)
        
        if n_validation > 0:
            precision = n_matches / n_detections if n_detections > 0 else 0
            recall = n_matches / n_validation
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1_score = np.nan
        
        validation_results = {
            'validation_available': True,
            'n_detections': n_detections,
            'n_validation_sources': n_validation,
            'n_matches': n_matches,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'matches': matches
        }
        
        logger.info(f"Validation: {n_matches}/{n_detections} detections matched, "
                   f"Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        return validation_results
    
    def _generate_quality_flags(self, super_emitters: pd.DataFrame, 
                              ds: xr.Dataset) -> Dict:
        """Generate quality flags for the detection results."""
        
        flags = {
            'data_quality': {
                'total_pixels': int(ds.ch4.count()),
                'valid_pixels': int((~np.isnan(ds.ch4)).sum()),
                'coverage_fraction': float((~np.isnan(ds.ch4)).sum() / ds.ch4.size),
                'temporal_coverage_days': len(ds.time)
            },
            'detection_quality': {
                'n_super_emitters': len(super_emitters),
                'mean_detection_score': float(super_emitters['detection_score'].mean()) if len(super_emitters) > 0 else 0,
                'mean_enhancement': float(super_emitters['mean_enhancement'].mean()) if len(super_emitters) > 0 else 0,
                'facility_association_rate': float(super_emitters['facility_id'].notna().mean()) if len(super_emitters) > 0 else 0
            },
            'spatial_distribution': {
                'lat_range': [float(ds.lat.min()), float(ds.lat.max())],
                'lon_range': [float(ds.lon.min()), float(ds.lon.max())],
                'emitter_lat_range': [float(super_emitters['center_lat'].min()), float(super_emitters['center_lat'].max())] if len(super_emitters) > 0 else [np.nan, np.nan],
                'emitter_lon_range': [float(super_emitters['center_lon'].min()), float(super_emitters['center_lon'].max())] if len(super_emitters) > 0 else [np.nan, np.nan]
            }
        }
        
        return flags
    
    def update_known_emitters(self, new_detections: pd.DataFrame):
        """Update the database of known super-emitters."""
        
        # This would implement logic to update the persistent database
        # For now, just store in memory
        if self.known_emitters is None:
            self.known_emitters = new_detections.copy()
        else:
            # Merge with existing, avoiding duplicates
            combined = pd.concat([self.known_emitters, new_detections], ignore_index=True)
            # Remove duplicates based on spatial proximity
            self.known_emitters = self._remove_duplicate_emitters(combined)
        
        logger.info(f"Updated known emitters database. Total: {len(self.known_emitters)}")
    
    def _remove_duplicate_emitters(self, emitters: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate emitters based on spatial proximity."""
        
        if len(emitters) <= 1:
            return emitters
        
        coords = emitters[['center_lat', 'center_lon']].values
        coords_km = self._geo_to_cartesian(coords[:, 0], coords[:, 1])
        
        eps_km = self.clustering_params['eps_km']
        dbscan = DBSCAN(eps=eps_km, min_samples=1)
        labels = dbscan.fit_predict(coords_km)
        
        # Keep the most recent detection for each cluster
        unique_emitters = []
        for label in np.unique(labels):
            group_mask = labels == label
            group = emitters[group_mask]
            
            # Keep the one with the latest detection
            if 'last_detected' in group.columns:
                latest = group.loc[group['last_detected'].idxmax()]
            else:
                latest = group.iloc[0]
            
            unique_emitters.append(latest)
        
        return pd.DataFrame(unique_emitters)
    
    def get_detection_summary(self) -> Dict:
        """Get summary statistics of recent detections."""
        
        if self.known_emitters is None or len(self.known_emitters) == 0:
            return {'total_emitters': 0}
        
        summary = {
            'total_emitters': len(self.known_emitters),
            'active_emitters': len(self.known_emitters[
                pd.to_datetime(self.known_emitters['last_detected']) > 
                (datetime.now() - timedelta(days=30))
            ]) if 'last_detected' in self.known_emitters.columns else 0,
            'mean_emission_rate': float(self.known_emitters['estimated_emission_rate_kg_hr'].mean()),
            'total_emission_rate': float(self.known_emitters['estimated_emission_rate_kg_hr'].sum()),
            'facility_associations': int(self.known_emitters['facility_id'].notna().sum()) if 'facility_id' in self.known_emitters.columns else 0
        }
        
        return summary