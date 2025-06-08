# ============================================================================
# FILE: tests/test_detector.py
# ============================================================================
import sys
from pathlib import Path

# FIXED: Correct path from tests/ to src/
sys.path.append(str(Path(__file__).parent.parent / "src/detection"))

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from  super_emitter_detector import SuperEmitterDetector

class TestSuperEmitterDetector:
    """Test super-emitter detection functionality."""
    
    @classmethod
    def setup_class(cls):
        """Setup test configuration and detector."""
        cls.config = {
            'super_emitters': {
                'detection': {
                    'enhancement_threshold': 50.0,
                    'emission_rate_threshold': 1000.0,
                    'spatial_extent_min': 4,
                    'persistence_days': 3,
                    'confidence_threshold': 0.7
                },
                'background': {
                    'method': 'rolling_percentile',
                    'percentile': 20,
                    'window_days': 30,
                    'spatial_radius_km': 100
                },
                'clustering': {
                    'algorithm': 'DBSCAN',
                    'eps_km': 5.0,
                    'min_samples': 3
                },
                'database': {
                    'facility_buffer_km': 2.0
                }
            },
            'tracking': {
                'persistence': {
                    'min_detections': 5,
                    'max_gap_days': 14
                }
            }
        }
        
        cls.detector = SuperEmitterDetector(cls.config)
    
    def test_detector_initialization(self):
        """Test that detector initializes correctly."""
        assert self.detector is not None
        assert hasattr(self.detector, 'config')
        assert hasattr(self.detector, 'detection_params')
        print("✅ Detector initialized successfully")
    
    def test_facility_database_loading(self):
        """Test facility database loading."""
        # Check if known facilities were loaded
        assert hasattr(self.detector, 'known_facilities')
        
        if len(self.detector.known_facilities) > 0:
            print(f"✅ Loaded {len(self.detector.known_facilities)} known facilities")
            print(f"Sample facility types: {self.detector.known_facilities['type'].unique()}")
        else:
            print("⚠️ No facilities loaded - using empty database")
    
    def create_mock_tropomi_dataset(self):
        """Create mock TROPOMI dataset for testing."""
        # Create a small grid
        lats = np.linspace(32.0, 33.0, 20)  # 20x20 grid
        lons = np.linspace(-102.0, -101.0, 20)
        times = pd.date_range('2023-06-01', periods=5, freq='D')
        
        # Create mock methane concentration data
        np.random.seed(42)
        ch4_data = np.random.normal(1850, 30, (len(times), len(lats), len(lons)))
        
        # Add some "hotspots" - elevated methane regions
        for t in range(len(times)):
            # Add a persistent hotspot
            ch4_data[t, 8:12, 8:12] += np.random.normal(100, 20, (4, 4))
            
            # Add a weaker, variable hotspot
            if t % 2 == 0:  # Only present on some days
                ch4_data[t, 15:18, 5:8] += np.random.normal(60, 15, (3, 3))
        
        # Quality assurance values (higher = better quality)
        qa_data = np.random.uniform(0.3, 1.0, (len(times), len(lats), len(lons)))
        
        # Create xarray dataset
        dataset = xr.Dataset({
            'ch4': (['time', 'lat', 'lon'], ch4_data),
            'qa_value': (['time', 'lat', 'lon'], qa_data)
        }, coords={
            'time': times,
            'lat': lats,
            'lon': lons
        })
        
        # Add attributes
        dataset['ch4'].attrs = {
            'long_name': 'CH4 column volume mixing ratio dry air',
            'units': 'ppb'
        }
        
        return dataset
    
    def test_mock_data_creation(self):
        """Test mock data creation."""
        dataset = self.create_mock_tropomi_dataset()
        
        assert dataset is not None
        assert 'ch4' in dataset.data_vars
        assert 'qa_value' in dataset.data_vars
        assert len(dataset.time) == 5
        
        print("✅ Mock dataset created successfully")
        print(f"Dataset shape: {dataset.dims}")
        print(f"CH4 range: {dataset.ch4.min().values:.1f} - {dataset.ch4.max().values:.1f} ppb")
    
    def test_background_calculation(self):
        """Test background calculation methods."""
        dataset = self.create_mock_tropomi_dataset()
        
        try:
            # Test the background calculation
            ds_with_bg = self.detector._calculate_background(dataset)
            
            assert 'background' in ds_with_bg.data_vars
            assert 'enhancement' in ds_with_bg.data_vars
            
            # Check that enhancement makes sense
            enhancement = ds_with_bg.enhancement
            assert not np.isnan(enhancement).all()
            
            print("✅ Background calculation successful")
            print(f"Enhancement range: {enhancement.min().values:.1f} - {enhancement.max().values:.1f} ppb")
            
            # Check if we detect the artificial hotspots
            max_enhancement = enhancement.max().values
            if max_enhancement > 50:  # Our artificial hotspots should show up
                print(f"✅ Artificial hotspots detected (max enhancement: {max_enhancement:.1f} ppb)")
            
        except Exception as e:
            pytest.fail(f"Background calculation failed: {e}")
    
    def test_statistical_detection(self):
        """Test statistical anomaly detection."""
        dataset = self.create_mock_tropomi_dataset()
        
        try:
            # Calculate background first
            ds_with_bg = self.detector._calculate_background(dataset)
            
            # Test statistical detection
            detected = self.detector._statistical_detection(ds_with_bg)
            
            assert 'detection_mask' in detected.data_vars
            assert 'detection_score' in detected.data_vars
            
            # Check if any detections were made
            detections = detected.detection_mask.sum().values
            detection_score = detected.detection_score.max().values
            
            print(f"✅ Statistical detection completed")
            print(f"Total detections: {detections}")
            print(f"Max detection score: {detection_score:.3f}")
            
            if detections > 0:
                print("✅ Successfully detected anomalies in mock data")
            
        except Exception as e:
            pytest.fail(f"Statistical detection failed: {e}")
    
    def test_full_detection_pipeline(self):
        """Test the full detection pipeline with mock data."""
        dataset = self.create_mock_tropomi_dataset()
        
        try:
            # Run full detection pipeline
            results = self.detector.detect_super_emitters(dataset)
            
            assert 'super_emitters' in results
            assert 'detection_metadata' in results
            
            super_emitters = results['super_emitters']
            
            print(f"✅ Full detection pipeline completed")
            print(f"Super-emitters found: {len(super_emitters)}")
            
            if len(super_emitters) > 0:
                print("Sample detection:")
                print(f"  Location: ({super_emitters.iloc[0]['center_lat']:.3f}, {super_emitters.iloc[0]['center_lon']:.3f})")
                print(f"  Enhancement: {super_emitters.iloc[0]['mean_enhancement']:.1f} ppb")
                print(f"  Emission rate: {super_emitters.iloc[0]['estimated_emission_rate_kg_hr']:.1f} kg/hr")
                print("✅ Pipeline successfully detects super-emitters in mock data")
            else:
                print("⚠️ No super-emitters detected - may need to adjust thresholds")
            
        except Exception as e:
            pytest.fail(f"Full detection pipeline failed: {e}")
    
    def test_facility_association(self):
        """Test association with known facilities."""
        # Create a small DataFrame to test facility association
        test_emitters = pd.DataFrame({
            'emitter_id': ['SE_0001', 'SE_0002'],
            'center_lat': [32.0, 32.5],  # These should be near our mock facilities
            'center_lon': [-102.5, -101.8],
            'estimated_emission_rate_kg_hr': [1200, 800],
            'facility_id': [None, None],
            'facility_name': [None, None],
            'facility_type': [None, None]
        })
        
        try:
            associated = self.detector._associate_with_facilities(test_emitters)
            
            print("✅ Facility association test completed")
            
            # Check if any associations were made
            associations = associated['facility_id'].notna().sum()
            print(f"Facility associations made: {associations}/{len(associated)}")
            
            if associations > 0:
                print("✅ Successfully associated emitters with facilities")
            
        except Exception as e:
            pytest.fail(f"Facility association failed: {e}")

if __name__ == "__main__":
    # Run tests manually
    test_class = TestSuperEmitterDetector()
    test_class.setup_class()
    
    print("🧪 Testing Super-Emitter Detection")
    print("=" * 50)
    
    try:
        test_class.test_detector_initialization()
        test_class.test_facility_database_loading()
        test_class.test_mock_data_creation()
        test_class.test_background_calculation()
        test_class.test_statistical_detection()
        test_class.test_full_detection_pipeline()
        test_class.test_facility_association()
        
        print("\n✅ All detector tests completed!")
        
    except Exception as e:
        print(f"\n❌ Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n💡 Next steps:")
    print("1. Run with real TROPOMI data using valid date ranges")
    print("2. Adjust detection thresholds based on real data characteristics")
    print("3. Test with known super-emitter regions (Permian Basin, etc.)")