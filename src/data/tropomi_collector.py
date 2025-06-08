# ============================================================================
# FILE: src/data/tropomi_collector.py - FIXED VERSION
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
            if self.gee_config.get('service_account_file'):
                credentials = ee.ServiceAccountCredentials(
                    email=None,
                    key_file=self.gee_config['service_account_file']
                )
                ee.Initialize(credentials)
            else:
                ee.Initialize(project=self.gee_config.get('project_id', None))
            
            logger.info("Google Earth Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Earth Engine: {e}")
            raise
    
    def create_region_geometry(self) -> ee.Geometry:
        """Create Earth Engine geometry from config."""
        roi = self.data_config['region_of_interest']
        
        if roi['type'] == 'bbox':
            coords = roi['coordinates']
            return ee.Geometry.Rectangle(coords)
        elif roi['type'] == 'polygon':
            return ee.Geometry.Polygon(roi['coordinates'])
        elif roi['type'] == 'global':
            return ee.Geometry.Rectangle([-180, -85, 180, 85])
        else:
            raise ValueError(f"Unsupported region type: {roi['type']}")
    
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
                geometry = self.create_region_geometry()
            
            # Get TROPOMI collection with correct band names
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
        """Get filtered TROPOMI image collection with correct bands."""
        
        collection_name = self.tropomi_config['collection']
        
        # Base collection
        collection = (ee.ImageCollection(collection_name)
                     .filterDate(start_date, end_date)
                     .filterBounds(geometry))
        
        # Select available bands (based on diagnostic output)
        # Primary methane data and uncertainty
        bands_to_select = [
            'CH4_column_volume_mixing_ratio_dry_air',
            'CH4_column_volume_mixing_ratio_dry_air_uncertainty'
        ]
        
        collection = collection.select(bands_to_select)
        
        # Apply quality filters using uncertainty as a proxy for quality
        collection = self._apply_quality_filters(collection)
        
        return collection
    
    def _apply_quality_filters(self, collection: ee.ImageCollection) -> ee.ImageCollection:
        """Apply quality filtering using uncertainty as quality measure."""
        
        def quality_filter(image):
            # Use uncertainty as quality measure
            # Lower uncertainty = higher quality
            uncertainty = image.select('CH4_column_volume_mixing_ratio_dry_air_uncertainty')
            ch4 = image.select('CH4_column_volume_mixing_ratio_dry_air')
            
            # Filter out very high uncertainties (poor quality)
            max_uncertainty = 50.0  # ppb - adjust based on data characteristics
            quality_mask = uncertainty.lt(max_uncertainty)
            
            # Filter out unrealistic CH4 values
            ch4_mask = ch4.gt(1600).And(ch4.lt(2500))  # Typical atmospheric range
            
            # Combine masks
            combined_mask = quality_mask.And(ch4_mask)
            
            return image.updateMask(combined_mask)
        
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
            'quality_filtering': 'uncertainty_based'
        })
        
        return combined_ds
    
    def _sample_image_regular_grid(self, image: ee.Image, bounds: list, 
                                  grid_size: int = 50) -> Optional[Dict]:
        """Sample image on a regular grid."""
        
        west, south, east, north = bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1]
        
        try:
            # Use sampleRectangle for smaller regions
            if (east - west) * (north - south) < 25:  # Small region - about 5x5 degrees
                sample = image.sampleRectangle(
                    region=ee.Geometry.Rectangle([west, south, east, north]),
                    defaultValue=-9999
                )
                return sample.getInfo()
            else:
                # For larger regions, use reduceRegion with a mean
                sample = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee.Geometry.Rectangle([west, south, east, north]),
                    scale=7000,  # TROPOMI native resolution ~7km
                    maxPixels=1e9
                )
                return sample.getInfo()
                
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
            uncertainty_data = props.get('CH4_column_volume_mixing_ratio_dry_air_uncertainty', [])
            
            if not ch4_data or not isinstance(ch4_data[0], list):
                raise ValueError("Invalid sample data format")
            
            ch4_array = np.array(ch4_data, dtype=float)
            uncertainty_array = np.array(uncertainty_data, dtype=float)
            
            # Replace missing values
            ch4_array[ch4_array == -9999] = np.nan
            uncertainty_array[uncertainty_array == -9999] = np.nan
            
            # Create coordinates
            height, width = ch4_array.shape
            west, south, east, north = bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1]
            
            lons = np.linspace(west, east, width)
            lats = np.linspace(north, south, height)  # Note: reversed for image coordinates
            
        else:
            # Single value from reduceRegion
            ch4_value = sample_data.get('CH4_column_volume_mixing_ratio_dry_air', np.nan)
            uncertainty_value = sample_data.get('CH4_column_volume_mixing_ratio_dry_air_uncertainty', np.nan)
            
            # Create single pixel dataset
            ch4_array = np.array([[ch4_value]])
            uncertainty_array = np.array([[uncertainty_value]])
            
            # Create single coordinate
            west, south, east, north = bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1]
            lons = np.array([(west + east) / 2])
            lats = np.array([(north + south) / 2])
        
        # Create Dataset with uncertainty-based quality measure
        ds = xr.Dataset({
            'ch4': (['lat', 'lon'], ch4_array),
            'ch4_uncertainty': (['lat', 'lon'], uncertainty_array),
            'qa_value': (['lat', 'lon'], 1.0 / (1.0 + uncertainty_array))  # Convert uncertainty to quality score
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
        
        ds['ch4_uncertainty'].attrs = {
            'long_name': 'CH4 column uncertainty',
            'units': 'ppb',
            'description': 'Uncertainty in CH4 measurements'
        }
        
        ds['qa_value'].attrs = {
            'long_name': 'Quality assurance value (derived from uncertainty)',
            'units': 'dimensionless',
            'description': 'Quality score: 1/(1+uncertainty), higher values = better quality'
        }
        
        return ds.expand_dims('time')
    
    def get_data_availability(self, start_date: str, end_date: str, 
                             region: Optional[Dict] = None) -> pd.DataFrame:
        """Check data availability for given period and region."""
        
        try:
            if region:
                geometry = self._create_geometry(region)
            else:
                geometry = self.create_region_geometry()
            
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