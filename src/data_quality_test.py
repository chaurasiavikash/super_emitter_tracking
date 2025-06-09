#!/usr/bin/env python3
"""
Debug script to investigate data quality issues in TROPOMI collection.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.tropomi_collector import TROPOMICollector
import yaml

def debug_data_quality():
    """Debug TROPOMI data quality issues."""
    
    print("🔍 DEBUGGING TROPOMI DATA QUALITY")
    print("="*50)
    
    # Load config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Try different regions known for methane emissions
    test_regions = [
        {
            'name': 'Permian Basin',
            'coordinates': [-104.0, 31.0, -102.0, 33.0]
        },
        {
            'name': 'Bakken',
            'coordinates': [-104.0, 47.0, -102.0, 49.0]
        },
        {
            'name': 'Four Corners',
            'coordinates': [-109.5, 36.5, -107.5, 37.5]
        }
    ]
    
    collector = TROPOMICollector(config)
    
    for region in test_regions:
        print(f"\n📍 Testing region: {region['name']}")
        
        # Override region
        config['data']['region_of_interest'] = {
            'type': 'bbox',
            'coordinates': region['coordinates']
        }
        
        # Test data availability first
        availability = collector.get_data_availability("2023-06-01", "2023-06-05")
        print(f"   Data availability: {len(availability)} time steps")
        
        if len(availability) > 0:
            print(f"   Date range: {availability['date'].min()} to {availability['date'].max()}")
            
            # Try to collect actual data
            try:
                dataset = collector.collect_data("2023-06-01", "2023-06-02")  # Just 2 days
                
                if dataset is not None:
                    print(f"   ✅ Data collected successfully!")
                    print(f"   Shape: {dataset.dims}")
                    
                    # Check data quality
                    ch4_data = dataset.ch4.values
                    valid_pixels = ~np.isnan(ch4_data)
                    
                    print(f"   Valid pixels: {valid_pixels.sum():,} / {ch4_data.size:,} ({valid_pixels.mean():.1%})")
                    
                    if valid_pixels.sum() > 0:
                        valid_ch4 = ch4_data[valid_pixels]
                        print(f"   CH4 range: {valid_ch4.min():.1f} - {valid_ch4.max():.1f} ppb")
                        print(f"   CH4 mean: {valid_ch4.mean():.1f} ± {valid_ch4.std():.1f} ppb")
                        
                        # Check for potential enhancements
                        background = np.nanpercentile(valid_ch4, 20)
                        enhancements = valid_ch4 - background
                        high_enhancements = enhancements[enhancements > 10]  # >10 ppb enhancement
                        
                        print(f"   Background (20th percentile): {background:.1f} ppb")
                        print(f"   Pixels with >10 ppb enhancement: {len(high_enhancements)}")
                        
                        if len(high_enhancements) > 0:
                            print(f"   ⚠️  POTENTIAL EMISSIONS DETECTED!")
                            print(f"   Max enhancement: {enhancements.max():.1f} ppb")
                        
                        # Create simple plot
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Plot CH4 concentrations
                        # Plot CH4 concentrations - fix for sparse data
                        ch4_2d = dataset.ch4.isel(time=0).values
                        valid_data = ch4_2d[~np.isnan(ch4_2d)]

                        if len(valid_data) > 0:
                            vmin, vmax = np.nanpercentile(valid_data, [5, 95])
                            im1 = ax1.imshow(ch4_2d, cmap='plasma', vmin=vmin, vmax=vmax)
                            ax1.set_title(f'CH4 Concentrations - {region["name"]} ({len(valid_data)} valid pixels)')
                        else:
                            im1 = ax1.imshow(ch4_2d, cmap='plasma')
                            ax1.set_title(f'CH4 Concentrations - {region["name"]} (No valid data)')
                        
 
                        plt.colorbar(im1, ax=ax1, label='CH4 (ppb)')
                        
                        # Plot enhancements
                        enhancement_2d = ch4_2d - np.nanpercentile(ch4_2d, 20)
                        im2 = ax2.imshow(enhancement_2d, cmap='hot', vmin=0, vmax=np.nanpercentile(enhancement_2d, 95))
                        ax2.set_title('CH4 Enhancements')
                        plt.colorbar(im2, ax=ax2, label='Enhancement (ppb)')
                        
                        plt.tight_layout()
                        plt.savefig(f'debug_{region["name"].lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
                        print(f"   📊 Plot saved: debug_{region['name'].lower().replace(' ', '_')}.png")
                        
                        return dataset  # Return first successful dataset for further analysis
                    else:
                        print(f"   ❌ No valid data pixels")
                else:
                    print(f"   ❌ Data collection failed")
                    
            except Exception as e:
                print(f"   ❌ Error collecting data: {e}")
        else:
            print(f"   ❌ No data available")
    
    print(f"\n🔧 RECOMMENDATIONS:")
    print(f"   1. Try different date ranges (some periods have better coverage)")
    print(f"   2. Use smaller regions with known emissions")
    print(f"   3. Further relax quality thresholds")
    print(f"   4. Check if your GEE project has access to TROPOMI data")

def test_detection_algorithm(dataset):
    """Test the detection algorithm with debug output."""
    if dataset is None:
        print("No dataset to test")
        return
    
    print(f"\n🔬 TESTING DETECTION ALGORITHM")
    print(f"="*40)
    
    # Load relaxed config
    with open('../config/config_relaxed.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    from detection.super_emitter_detector import SuperEmitterDetector
    
    detector = SuperEmitterDetector(config)
    
    try:
        results = detector.detect_super_emitters(dataset)
        
        print(f"Detection results:")
        print(f"  Super-emitters found: {len(results['super_emitters'])}")
        print(f"  Quality flags: {results['quality_flags']}")
        
        if len(results['super_emitters']) > 0:
            print(f"  🎉 SUCCESS! Found super-emitters with relaxed settings")
        else:
            print(f"  ❌ Still no detections. Need to debug further.")
            
    except Exception as e:
        print(f"Detection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    dataset = debug_data_quality()
    test_detection_algorithm(dataset)