# ============================================================================
# FILE: src/data/tropomi_collector.py
# ============================================================================
import ee
import numpy as np
import xarray as xr
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TROPOMICollector:
    """
    Collect and process TROPOMI methane data from Google Earth Engine.
    
    Features:
    - Efficient data collection with quality filtering
    - Geographic and temporal subsetting
    - Automatic retry and error handling
    - Data validation and preprocessing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.gee_config = config['gee']
        self.data_config = config['data']
        self.tropomi_config = config['data']['tropomi']
        
        self._initialize_gee()
        
    def _initialize_gee(self):
            
            """Initialize Google Earth Engine connection."""
            try:
                project_id = self.gee_config.get('project_id')
                service_account_file = self.gee_config.get('service_account_file')
                if service_account_file:
                    credentials = ee.ServiceAccountCredentials(
                        email=None,
                        key_file=service_account_file
                    )
                    if project_id:
                        ee.Initialize(credentials, project=project_id)
                    else:
                        ee.Initialize(credentials)
                else:
                    if project_id:
                        ee.Initialize(project=project_id)
                    else:
                        ee.Initialize()
                
                logger.info("Google Earth Engine initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Google Earth Engine: {e}")
                raise    
    
    def collect_data(self, start_date: str, end_date: str, 
                    region: Optional[Dict] = None) -> Optional[xr.Dataset]:
        """
        Collect TROPOMI methane data for specified period and region.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            region: Optional region override
            
        Returns:
            xarray Dataset with TROPOMI data or None if no data found
        """
        logger.info(f"Collecting TROPOMI data: {start_date} to {end_date}")
        
        try:
            # Get region geometry
            if region:
                geometry = self._create_geometry(region)
            else:
                geometry = self._create_geometry(self.data_config['region_of_interest'])
            
            # Get TROPOMI collection
            collection = self._get_tropomi_collection(start_date, end_date, geometry)
            
            # Check if data is available
            collection_info = collection.getInfo()
            
            if not collection_info or len(collection_info.get('features', [])) == 0:
                logger.warning(f"No TROPOMI data found for period {start_date} to {end_date}")
                return None
            
            logger.info(f"Found {len(collection_info['features'])} TROPOMI images")
            
            # Process collection to xarray Dataset
            dataset = self._collection_to_xarray(collection, geometry)
            
            if dataset is not None:
                logger.info(f"Successfully collected data with shape: {dataset.dims}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error collecting TROPOMI data: {e}")
            raise
    
    def _create_geometry(self, region_config: Dict) -> ee.Geometry:
        """Create Earth Engine geometry from region configuration."""
        
        region_type = region_config['type']
        
        if region_type == 'bbox':
            coords = region_config['coordinates']  # [west, south, east, north]
            return ee.Geometry.Rectangle(coords)
            
        elif region_type == 'polygon':
            return ee.Geometry.Polygon(region_config['coordinates'])
            
        elif region_type == 'global':
            return ee.Geometry.Rectangle([-180, -85, 180, 85])
            
        else:
            raise ValueError(f"Unsupported region type: {region_type}")
    
    def _get_tropomi_collection(self, start_date: str, end_date: str, 
                               geometry: ee.Geometry) -> ee.ImageCollection:
        """Get filtered TROPOMI image collection."""
        
        collection_name = self.tropomi_config['collection']
        
        # Base collection
        collection = (ee.ImageCollection(collection_name)
                     .filterDate(start_date, end_date)
                     .filterBounds(geometry))
        
        # Apply quality filters
        collection = self._apply_quality_filters(collection)
        
        # Select bands
        bands = ['CH4_column_volume_mixing_ratio_dry_air', 'qa_value']
        if 'cloud_fraction' in self.tropomi_config:
            bands.append('cloud_fraction')
        
        collection = collection.select(bands)
        
        return collection
    
    def _apply_quality_filters(self, collection: ee.ImageCollection) -> ee.ImageCollection:
        """Apply quality filtering to TROPOMI collection."""
        
        def quality_filter(image):
            # Quality mask
            qa = image.select('qa_value')
            quality_mask = qa.gte(self.tropomi_config['quality_threshold'])
            
            # Cloud mask if available
            if 'cloud_fraction_max' in self.tropomi_config:
                try:
                    cloud_fraction = image.select('cloud_fraction')
                    cloud_mask = cloud_fraction.lte(self.tropomi_config['cloud_fraction_max'])
                    quality_mask = quality_mask.And(cloud_mask)
                except:
                    pass  # Cloud fraction not available in all products
            
            return image.updateMask(quality_mask)
        
        return collection.map(quality_filter)
    
    def _collection_to_xarray(self, collection: ee.ImageCollection, 
                             geometry: ee.Geometry) -> Optional[xr.Dataset]:
        """Convert Earth Engine ImageCollection to xarray Dataset."""
        
        # Get collection as list
        collection_list = collection.getInfo()
        
        if not collection_list['features']:
            return None
        
        datasets = []
        
        for feature in collection_list['features']:
            try:
                # Get image
                image_id = feature['id']
                image = ee.Image(image_id)
                
                # Get timestamp
                timestamp_ms = feature['properties']['system:time_start']
                timestamp = pd.to_datetime(timestamp_ms, unit='ms')
                
                # Sample image over region
                # For large regions, we might need to use a different approach
                bounds = geometry.bounds().getInfo()['coordinates'][0]
                
                # Create a regular grid for sampling
                sample_data = self._sample_image_regular_grid(image, bounds)
                
                if sample_data is not None:
                    # Create xarray Dataset
                    ds = self._create_xarray_dataset(sample_data, timestamp, bounds)
                    datasets.append(ds)
                    
            except Exception as e:
                logger.warning(f"Failed to process image {feature['id']}: {e}")
                continue
        
        if not datasets:
            logger.error("No valid datasets processed")
            return None
        
        # Combine all datasets along time dimension
        combined_ds = xr.concat(datasets, dim='time')
        
        # Add metadata
        combined_ds.attrs.update({
            'source': 'TROPOMI/Sentinel-5P',
            'collection': self.tropomi_config['collection'],
            'created': pd.Timestamp.now().isoformat(),
            'quality_threshold': self.tropomi_config['quality_threshold']
        })
        
        return combined_ds
    
    def _sample_image_regular_grid(self, image: ee.Image, bounds: list, 
                                  grid_size: int = 100) -> Optional[Dict]:
        """Sample image on a regular grid."""
        
        west, south, east, north = bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1]
        
        # Create regular grid
        lon_step = (east - west) / grid_size
        lat_step = (north - south) / grid_size
        
        try:
            # Use reduceRegion for smaller areas or sampleRectangle for larger ones
            if (east - west) * (north - south) < 100:  # Small region
                sample = image.sampleRectangle(
                    region=ee.Geometry.Rectangle([west, south, east, north]),
                    defaultValue=-9999
                )
                return sample.getInfo()
            else:
                # For larger regions, sample at points
                points = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        lon = west + i * lon_step
                        lat = south + j * lat_step
                        points.append([lon, lat])
                
                # Limit number of points for API limits
                if len(points) > 1000:
                    points = points[::len(points)//1000]
                
                point_collection = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point(point)) for point in points
                ])
                
                sampled = image.sampleRegions(
                    collection=point_collection,
                    scale=7000,  # TROPOMI resolution
                    tileScale=4
                )
                
                return sampled.getInfo()
                
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            return None
    
    def _create_xarray_dataset(self, sample_data: Dict, timestamp: pd.Timestamp, 
                              bounds: list) -> xr.Dataset:
        """Create xarray Dataset from sampled data."""
        
        if 'properties' in sample_data:
            # Rectangle sampling
            props = sample_data['properties']
            
            ch4_data = props.get('CH4_column_volume_mixing_ratio_dry_air', [])
            qa_data = props.get('qa_value', [])
            
            if not ch4_data or not isinstance(ch4_data[0], list):
                raise ValueError("Invalid sample data format")
            
            ch4_array = np.array(ch4_data, dtype=float)
            qa_array = np.array(qa_data, dtype=float)
            
            # Replace missing values
            ch4_array[ch4_array == -9999] = np.nan
            qa_array[qa_array == -9999] = np.nan
            
            # Create coordinates
            height, width = ch4_array.shape
            west, south, east, north = bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1]
            
            lons = np.linspace(west, east, width)
            lats = np.linspace(north, south, height)  # Note: reversed for image coordinates
            
        else:
            # Point sampling
            features = sample_data.get('features', [])
            
            if not features:
                raise ValueError("No sample features found")
            
            # Extract data from features
            lats, lons, ch4_values, qa_values = [], [], [], []
            
            for feature in features:
                geom = feature['geometry']['coordinates']
                props = feature['properties']
                
                lons.append(geom[0])
                lats.append(geom[1])
                ch4_values.append(props.get('CH4_column_volume_mixing_ratio_dry_air', np.nan))
                qa_values.append(props.get('qa_value', np.nan))
            
            # Convert to regular grid (simple interpolation)
            # This is a simplified approach - in practice you might want more sophisticated gridding
            lats = np.array(lats)
            lons = np.array(lons)
            ch4_values = np.array(ch4_values)
            qa_values = np.array(qa_values)
            
            # Create regular grid
            grid_size = int(np.sqrt(len(lats)))
            if grid_size < 10:
                grid_size = 10
            
            lat_grid = np.linspace(lats.min(), lats.max(), grid_size)
            lon_grid = np.linspace(lons.min(), lons.max(), grid_size)
            
            # Simple nearest neighbor gridding
            ch4_array = np.full((grid_size, grid_size), np.nan)
            qa_array = np.full((grid_size, grid_size), np.nan)
            
            for i, lat in enumerate(lat_grid):
                for j, lon in enumerate(lon_grid):
                    # Find nearest point
                    distances = (lats - lat)**2 + (lons - lon)**2
                    nearest_idx = np.argmin(distances)
                    
                    ch4_array[i, j] = ch4_values[nearest_idx]
                    qa_array[i, j] = qa_values[nearest_idx]
            
            lats = lat_grid
            lons = lon_grid
        
        # Create Dataset
        ds = xr.Dataset({
            'ch4': (['lat', 'lon'], ch4_array),
            'qa_value': (['lat', 'lon'], qa_array)
        }, coords={
            'lat': lats,
            'lon': lons,
            'time': timestamp
        })
        
        # Add variable attributes
        ds['ch4'].attrs = {
            'long_name': 'CH4 column volume mixing ratio dry air',
            'units': 'ppb',
            'source': 'TROPOMI'
        }
        
        ds['qa_value'].attrs = {
            'long_name': 'Quality assurance value',
            'units': 'dimensionless',
            'valid_range': [0, 1]
        }
        
        return ds.expand_dims('time')
    
    def get_data_availability(self, start_date: str, end_date: str, 
                             region: Optional[Dict] = None) -> pd.DataFrame:
        """Check data availability for given period and region."""
        
        try:
            if region:
                geometry = self._create_geometry(region)
            else:
                geometry = self._create_geometry(self.data_config['region_of_interest'])
            
            collection = self._get_tropomi_collection(start_date, end_date, geometry)
            
            # Get collection info
            collection_list = collection.getInfo()
            
            if not collection_list['features']:
                return pd.DataFrame()
            
            # Extract availability info
            availability_data = []
            
            for feature in collection_list['features']:
                timestamp_ms = feature['properties']['system:time_start']
                timestamp = pd.to_datetime(timestamp_ms, unit='ms')
                
                availability_data.append({
                    'date': timestamp.date(),
                    'datetime': timestamp,
                    'image_id': feature['id'],
                    'available': True
                })
            
            df = pd.DataFrame(availability_data)
            
            if not df.empty:
                df = df.sort_values('datetime')
                logger.info(f"Data available for {len(df)} time steps")
            
            return df
            
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            return pd.DataFrame()
    
    def test_connection(self) -> bool:
        """Test Google Earth Engine connection and data access."""
        
        try:
            # Try to access a small test collection
            test_collection = ee.ImageCollection(self.tropomi_config['collection']).limit(1)
            test_info = test_collection.getInfo()
            
            if test_info and test_info.get('features'):
                logger.info("TROPOMI data access test successful")
                return True
            else:
                logger.warning("TROPOMI collection appears to be empty")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False