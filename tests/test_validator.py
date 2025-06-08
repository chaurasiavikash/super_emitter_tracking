#!/usr/bin/env python3
# ============================================================================
# FILE: tests/test_real_data_pipeline.py
# Test the pipeline with real TROPOMI data
# ============================================================================

import sys
from pathlib import Path

# Add src to path (from tests folder, go up one level to project root, then into src)
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from data.tropomi_collector import TROPOMICollector
from detection.super_emitter_detector import SuperEmitterDetector

def test_real_data_pipeline():
    """Test the detection pipeline with real TROPOMI data."""
    
    print("🛰️ Testing Pipeline with Real TROPOMI Data")
    print("=" * 60)
    
    # Configuration
    config = {
        'gee': {
            'project_id': 'sodium-lore-456715-i3',
            'service_account_file': None
        },
        'data': {
            'region_of_interest': {
                'type': 'bbox',
                'coordinates': [-103.0, 31.5, -101.5, 33.0]  # Permian Basin
            },
            'tropomi': {
                'collection': 'COPERNICUS/S5P/OFFL/L3_CH4',
                'quality_threshold': 0.5
            }
        },
        'super_emitters': {
            'detection': {
                'enhancement_threshold': 30.0,  # Lower threshold for real data
                'emission_rate_threshold': 500.0,  # Lower threshold for testing
                'confidence_threshold': 0.6
            },
            'background': {
                'method': 'rolling_percentile',
                'percentile': 20,
                'window_days': 7
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
                'min_detections': 3,  # Lower for testing
                'max_gap_days': 7
            }
        }
    }
    
    try:
        # Step 1: Collect real data
        print("📡 Step 1: Collecting real TROPOMI data...")
        collector = TROPOMICollector(config)
        dataset = collector.collect_data('2023-06-01', '2023-06-03')  # Short period for testing
        
        if dataset is None:
            print("❌ No data collected")
            return
        
        print(f"✅ Collected dataset: {dataset.dims}")
        print(f"CH4 range: {dataset.ch4.min().values:.1f} - {dataset.ch4.max().values:.1f} ppb")
        print(f"Time steps: {len(dataset.time)}")
        
        # Step 2: Analyze data characteristics
        print("\n🔍 Step 2: Analyzing data characteristics...")
        
        # Calculate basic statistics
        ch4_mean = float(dataset.ch4.mean())
        ch4_std = float(dataset.ch4.std())
        valid_pixels = (~np.isnan(dataset.ch4)).sum()
        total_pixels = dataset.ch4.size
        
        print(f"Mean CH4: {ch4_mean:.1f} ± {ch4_std:.1f} ppb")
        print(f"Valid pixels: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels:.1%})")
        
        # Check for potential hotspots (simple analysis)
        threshold = ch4_mean + 2 * ch4_std
        potential_hotspots = (dataset.ch4 > threshold).sum()
        print(f"Potential hotspots (>2σ): {potential_hotspots} pixels")
        
        # Step 3: Test detection pipeline
        print("\n🔍 Step 3: Testing super-emitter detection...")
        detector = SuperEmitterDetector(config)
        
        try:
            results = detector.detect_super_emitters(dataset)
            
            super_emitters = results['super_emitters']
            print(f"✅ Detection completed!")
            print(f"Super-emitters detected: {len(super_emitters)}")
            
            if len(super_emitters) > 0:
                print("\n📊 Detection Results:")
                for i, emitter in super_emitters.iterrows():
                    print(f"  {emitter['emitter_id']}:")
                    print(f"    Location: ({emitter['center_lat']:.3f}°N, {emitter['center_lon']:.3f}°W)")
                    print(f"    Enhancement: {emitter['mean_enhancement']:.1f} ppb")
                    print(f"    Emission rate: {emitter['estimated_emission_rate_kg_hr']:.0f} kg/hr")
                    print(f"    Confidence: {emitter['detection_score']:.2f}")
                    if emitter.get('facility_name'):
                        print(f"    Associated facility: {emitter['facility_name']}")
                    print()
                
                # Step 4: Create simple visualization
                print("📊 Step 4: Creating visualization...")
                create_simple_plot(dataset, super_emitters)
                
            else:
                print("⚠️ No super-emitters detected with current thresholds")
                print("💡 Try lowering detection thresholds:")
                print("   - enhancement_threshold: 20.0 ppb")
                print("   - emission_rate_threshold: 300.0 kg/hr")
                print("   - confidence_threshold: 0.5")
                
                # Show enhancement distribution anyway
                print("\n📈 Enhancement distribution:")
                create_diagnostic_plot(dataset)
                
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

def create_simple_plot(dataset, super_emitters):
    """Create a simple plot of the data and detections."""
    
    try:
        # Take first time step for plotting
        ch4_data = dataset.ch4.isel(time=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: CH4 concentrations
        im1 = ax1.imshow(ch4_data, cmap='viridis', aspect='auto')
        ax1.set_title('CH4 Concentrations (ppb)')
        ax1.set_xlabel('Longitude Index')
        ax1.set_ylabel('Latitude Index')
        plt.colorbar(im1, ax=ax1, label='CH4 (ppb)')
        
        # Plot 2: Enhancement (if available)
        if 'enhancement' in dataset.data_vars:
            enh_data = dataset.enhancement.isel(time=0)
            im2 = ax2.imshow(enh_data, cmap='plasma', aspect='auto')
            ax2.set_title('CH4 Enhancement (ppb)')
            plt.colorbar(im2, ax=ax2, label='Enhancement (ppb)')
        else:
            ax2.text(0.5, 0.5, 'Enhancement not calculated', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Enhancement (not available)')
        
        # Add super-emitter locations
        if len(super_emitters) > 0:
            # Convert lat/lon to pixel indices (approximate)
            lats = dataset.lat.values
            lons = dataset.lon.values
            
            for _, emitter in super_emitters.iterrows():
                # Find closest pixel
                lat_idx = np.argmin(np.abs(lats - emitter['center_lat']))
                lon_idx = np.argmin(np.abs(lons - emitter['center_lon']))
                
                # Mark on both plots
                ax1.plot(lon_idx, lat_idx, 'r*', markersize=15, markeredgecolor='white')
                ax2.plot(lon_idx, lat_idx, 'r*', markersize=15, markeredgecolor='white')
        
        plt.tight_layout()
        
        # Save to tests folder
        output_file = Path(__file__).parent / 'tropomi_test_results.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved as '{output_file}'")
        
        # Show plot if possible
        try:
            plt.show()
        except:
            print("(Plot display not available - saved to file)")
            
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

def create_diagnostic_plot(dataset):
    """Create diagnostic plots even when no super-emitters are detected."""
    
    try:
        # Basic statistics and histograms
        ch4_data = dataset.ch4.values.flatten()
        ch4_data = ch4_data[~np.isnan(ch4_data)]
        
        if len(ch4_data) == 0:
            print("No valid CH4 data for plotting")
            return
        
        print(f"CH4 statistics:")
        print(f"  Mean: {ch4_data.mean():.1f} ppb")
        print(f"  Std: {ch4_data.std():.1f} ppb")
        print(f"  Min: {ch4_data.min():.1f} ppb")
        print(f"  Max: {ch4_data.max():.1f} ppb")
        print(f"  95th percentile: {np.percentile(ch4_data, 95):.1f} ppb")
        
        # Create histogram
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(ch4_data, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('CH4 Concentration (ppb)')
        ax.set_ylabel('Frequency')
        ax.set_title('CH4 Concentration Distribution')
        ax.axvline(ch4_data.mean(), color='red', linestyle='--', label=f'Mean: {ch4_data.mean():.1f}')
        ax.axvline(np.percentile(ch4_data, 95), color='orange', linestyle='--', label=f'95th %ile: {np.percentile(ch4_data, 95):.1f}')
        ax.legend()
        
        plt.tight_layout()
        
        # Save to tests folder
        output_file = Path(__file__).parent / 'tropomi_diagnostic.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Diagnostic plot saved as '{output_file}'")
        
    except Exception as e:
        print(f"⚠️ Diagnostic plotting failed: {e}")

if __name__ == "__main__":
    test_real_data_pipeline()