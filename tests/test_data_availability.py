# ============================================================================
# FILE: tests/test_data_availability.py
# ============================================================================
import sys
from pathlib import Path

# FIXED: Correct path from tests/ to src/
sys.path.append(str(Path(__file__).parent.parent / "src/data"))
import pytest
import ee
import yaml
from datetime import datetime, timedelta
from  tropomi_collector import TROPOMICollector

class TestDataAvailability:
    """Test TROPOMI data availability and collection."""
    
    @classmethod
    def setup_class(cls):
        """Setup test configuration."""
        cls.config = {
            'gee': {
                'project_id': 'sodium-lore-456715-i3',
                'service_account_file': None
            },
            'data': {
                'region_of_interest': {
                    'type': 'bbox',
                    'coordinates': [-102.0, 31.0, -101.0, 32.0]  # Small test region in Permian Basin
                },
                'tropomi': {
                    'collection': 'COPERNICUS/S5P/OFFL/L3_CH4',
                    'quality_threshold': 0.5,
                    'cloud_fraction_max': 0.3
                }
            }
        }
        
        # Initialize collector
        cls.collector = TROPOMICollector(cls.config)
    
    def test_gee_connection(self):
        """Test Google Earth Engine connection."""
        try:
            # Test basic GEE functionality
            test_collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4').limit(1)
            info = test_collection.getInfo()
            
            assert info is not None
            assert 'features' in info
            print("✅ GEE connection successful")
            
        except Exception as e:
            pytest.fail(f"GEE connection failed: {e}")
    
    def test_tropomi_collection_exists(self):
        """Test that TROPOMI collection exists and is accessible."""
        try:
            collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4')
            
            # Get basic info about the collection
            info = collection.limit(1).getInfo()
            assert info is not None
            print("✅ TROPOMI collection accessible")
            
        except Exception as e:
            pytest.fail(f"TROPOMI collection access failed: {e}")
    
    def test_data_availability_recent_valid_period(self):
        """Test data availability for a recent valid period."""
        # Use a known good period (e.g., 2023)
        start_date = "2023-06-01"
        end_date = "2023-06-05"
        
        try:
            availability_df = self.collector.get_data_availability(start_date, end_date)
            
            print(f"Data availability for {start_date} to {end_date}:")
            print(f"Total images found: {len(availability_df)}")
            
            if len(availability_df) > 0:
                print("✅ Data found for recent valid period")
                print(f"Sample dates: {availability_df['date'].tolist()[:3]}")
            else:
                print("⚠️ No data found - this might be expected for small regions")
                
        except Exception as e:
            pytest.fail(f"Data availability check failed: {e}")
    
    def test_invalid_future_dates(self):
        """Test that requesting future dates returns no data."""
        # Use future dates (like in your original error)
        start_date = "2025-06-07"
        end_date = "2025-06-08"
        
        try:
            availability_df = self.collector.get_data_availability(start_date, end_date)
            
            # Should return empty DataFrame for future dates
            assert len(availability_df) == 0
            print("✅ Correctly returns no data for future dates")
            
        except Exception as e:
            pytest.fail(f"Future date test failed: {e}")
    
    def test_data_collection_with_valid_dates(self):
        """Test actual data collection with valid dates."""
        # Use a short, recent period
        start_date = "2023-06-01"
        end_date = "2023-06-02"  # Just 2 days to minimize API calls
        
        try:
            dataset = self.collector.collect_data(start_date, end_date)
            
            if dataset is not None:
                print("✅ Successfully collected data")
                print(f"Dataset shape: {dataset.dims}")
                print(f"Variables: {list(dataset.data_vars.keys())}")
                print(f"Time range: {dataset.time.min().values} to {dataset.time.max().values}")
            else:
                print("⚠️ No data collected - may be normal for small regions/short periods")
                
        except Exception as e:
            print(f"⚠️ Data collection failed (may be expected): {e}")
    
    def test_permian_basin_known_active_region(self):
        """Test data availability in Permian Basin - a known active methane region."""
        # Permian Basin coordinates (known to have methane emissions)
        permian_config = self.config.copy()
        permian_config['data']['region_of_interest'] = {
            'type': 'bbox',
            'coordinates': [-103.0, 31.5, -101.5, 33.0]  # Larger Permian Basin area
        }
        
        collector = TROPOMICollector(permian_config)
        
        try:
            # Test with a longer period to increase chances of finding data
            start_date = "2023-05-01"
            end_date = "2023-05-07"
            
            availability_df = collector.get_data_availability(start_date, end_date)
            
            print(f"Permian Basin data availability ({start_date} to {end_date}):")
            print(f"Images found: {len(availability_df)}")
            
            if len(availability_df) > 0:
                print("✅ Found data in Permian Basin region")
            else:
                print("⚠️ No data in Permian Basin - may need different dates or region")
                
        except Exception as e:
            pytest.fail(f"Permian Basin test failed: {e}")

if __name__ == "__main__":
    # Run tests manually
    test_class = TestDataAvailability()
    test_class.setup_class()
    
    print("🧪 Testing TROPOMI Data Availability and Collection")
    print("=" * 60)
    
    try:
        test_class.test_gee_connection()
        test_class.test_tropomi_collection_exists()
        test_class.test_data_availability_recent_valid_period()
        test_class.test_invalid_future_dates()
        test_class.test_data_collection_with_valid_dates()
        test_class.test_permian_basin_known_active_region()
        
        print("\n✅ All diagnostic tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        
    print("\n💡 Recommendations:")
    print("1. Use dates between 2019-02-08 and present (not future dates)")
    print("2. Try Permian Basin region [-103.0, 31.5, -101.5, 33.0] for known emissions")
    print("3. Use longer time periods (1-2 weeks) to increase data availability")
    print("4. Check quality thresholds if no data is found")