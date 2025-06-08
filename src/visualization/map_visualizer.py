# ============================================================================
# FILE: src/visualization/map_plotter.py
# ============================================================================
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
import geopandas as gpd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import xarray as xr

logger = logging.getLogger(__name__)

class MethaneMapPlotter:
    """
    Advanced visualization tools for methane super-emitter mapping and analysis.
    
    Features:
    - Interactive and static maps
    - Time series visualization
    - Emission enhancement maps
    - Multi-layer visualizations
    - Export capabilities
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.map_config = self.viz_config.get('maps', {})
        
    def plot_enhancement_map(self, dataset: xr.Dataset, time_idx: int = 0,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot methane enhancement map using Cartopy."""
        
        logger.info(f"Creating enhancement map for time index {time_idx}")
        
        # Extract data for specified time
        if 'enhancement' in dataset.data_vars:
            enhancement = dataset.enhancement.isel(time=time_idx)
        else:
            enhancement = dataset.ch4.isel(time=time_idx) - dataset.ch4.isel(time=time_idx).median()
        
        lats = dataset.lat.values
        lons = dataset.lon.values
        
        # Create figure with Cartopy projection
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Plot enhancement data
        enhancement_data = enhancement.values
        valid_mask = ~np.isnan(enhancement_data)
        
        if np.any(valid_mask):
            # Create meshgrid
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            # Plot enhancement with custom colormap
            im = ax.pcolormesh(
                lon_grid, lat_grid, enhancement_data,
                transform=ccrs.PlateCarree(),
                cmap='plasma',
                vmin=np.nanpercentile(enhancement_data, 5),
                vmax=np.nanpercentile(enhancement_data, 95),
                alpha=0.8
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('CH₄ Enhancement (ppb)', fontsize=12)
        
        # Set extent and labels
        ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], 
                     crs=ccrs.PlateCarree())
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        # Add title
        timestamp = pd.to_datetime(dataset.time.values[time_idx])
        plt.title(f'Methane Enhancement Map\n{timestamp.strftime("%Y-%m-%d %H:%M")}', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Enhancement map saved to {save_path}")
        
        return fig
    
    def plot_hotspots_map(self, dataset: xr.Dataset, emission_results: pd.DataFrame,
                         time_idx: int = 0, save_path: Optional[str] = None) -> plt.Figure:
        """Plot detected hotspots on enhancement background."""
        
        logger.info(f"Creating hotspots map for time index {time_idx}")
        
        # Start with enhancement map
        fig = self.plot_enhancement_map(dataset, time_idx)
        ax = fig.get_axes()[0]
        
        if len(emission_results) > 0:
            # Plot hotspots as scatter points
            scatter = ax.scatter(
                emission_results['center_lon'],
                emission_results['center_lat'],
                c=emission_results['estimated_emission_rate_kg_hr'],
                s=emission_results['estimated_emission_rate_kg_hr'] / 20,  # Size by emission rate
                cmap='Reds',
                alpha=0.8,
                edgecolors='black',
                linewidth=1,
                transform=ccrs.PlateCarree(),
                zorder=10
            )
            
            # Add separate colorbar for emissions
            cbar2 = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.12)
            cbar2.set_label('Emission Rate (kg/hr)', fontsize=12)
            
            # Add annotations for largest emitters
            top_emitters = emission_results.nlargest(5, 'estimated_emission_rate_kg_hr')
            for _, emitter in top_emitters.iterrows():
                ax.annotate(
                    f"{emitter['estimated_emission_rate_kg_hr']:.0f} kg/hr",
                    xy=(emitter['center_lon'], emitter['center_lat']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    transform=ccrs.PlateCarree()
                )
        
        plt.title(f'Methane Super-Emitters\n{len(emission_results)} hotspots detected', 
                 fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hotspots map saved to {save_path}")
        
        return fig
    
    def plot_time_series(self, emission_results: pd.DataFrame,
                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot time series of emissions."""
        
        logger.info("Creating emission time series plot")
        
        if len(emission_results) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No emission data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            return fig
        
        # Group by time if time column exists
        if 'time' in emission_results.columns:
            time_series = emission_results.groupby('time')['estimated_emission_rate_kg_hr'].sum()
        else:
            # Create mock time series from emitter IDs
            time_series = emission_results.groupby('emitter_id')['estimated_emission_rate_kg_hr'].first()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Total emissions over time
        ax1.plot(time_series.index, time_series.values, 
                marker='o', linewidth=2, markersize=6, color='red')
        ax1.set_title('Total Methane Emissions Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Emission Rate (kg/hr)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual emitter contributions
        if len(emission_results) > 1:
            top_emitters = emission_results.nlargest(10, 'estimated_emission_rate_kg_hr')
            
            bars = ax2.bar(range(len(top_emitters)), 
                          top_emitters['estimated_emission_rate_kg_hr'],
                          color='orange', alpha=0.7)
            
            ax2.set_title('Top 10 Super-Emitters', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Emission Rate (kg/hr)', fontsize=12)
            ax2.set_xlabel('Emitter Rank', fontsize=12)
            ax2.set_xticks(range(len(top_emitters)))
            ax2.set_xticklabels([f'#{i+1}' for i in range(len(top_emitters))])
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series plot saved to {save_path}")
        
        return fig
    
    def create_interactive_map(self, dataset: xr.Dataset, emission_results: pd.DataFrame,
                             save_path: Optional[str] = None) -> folium.Map:
        """Create interactive Folium map."""
        
        logger.info("Creating interactive Folium map")
        
        # Calculate map center
        if len(emission_results) > 0:
            center_lat = emission_results['center_lat'].mean()
            center_lon = emission_results['center_lon'].mean()
        else:
            center_lat = dataset.lat.mean().item()
            center_lon = dataset.lon.mean().item()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        # Add enhancement layer as heatmap if data available
        if 'enhancement' in dataset.data_vars and len(dataset.time) > 0:
            enhancement_data = dataset.enhancement.isel(time=0).values
            lats = dataset.lat.values
            lons = dataset.lon.values
            
            # Prepare heatmap data
            heat_data = []
            for i in range(len(lats)):
                for j in range(len(lons)):
                    if not np.isnan(enhancement_data[i, j]) and enhancement_data[i, j] > 0:
                        heat_data.append([lats[i], lons[j], float(enhancement_data[i, j])])
            
            if heat_data:
                HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
        
        # Add super-emitter markers
        if len(emission_results) > 0:
            for _, emitter in emission_results.iterrows():
                # Color based on emission rate
                emission_rate = emitter['estimated_emission_rate_kg_hr']
                if emission_rate > 2000:
                    color = 'red'
                    icon = 'exclamation-triangle'
                elif emission_rate > 1000:
                    color = 'orange'
                    icon = 'fire'
                else:
                    color = 'yellow'
                    icon = 'industry'
                
                # Create popup text
                popup_text = f"""
                <b>Super-Emitter</b><br>
                <b>ID:</b> {emitter.get('emitter_id', 'Unknown')}<br>
                <b>Emission Rate:</b> {emission_rate:.0f} kg/hr<br>
                <b>Enhancement:</b> {emitter.get('mean_enhancement', 0):.1f} ppb<br>
                <b>Detection Score:</b> {emitter.get('detection_score', 0):.2f}<br>
                <b>Location:</b> {emitter['center_lat']:.3f}°N, {emitter['center_lon']:.3f}°W
                """
                
                if 'facility_name' in emitter and pd.notna(emitter['facility_name']):
                    popup_text += f"<br><b>Facility:</b> {emitter['facility_name']}"
                
                folium.Marker(
                    location=[emitter['center_lat'], emitter['center_lon']],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"Emission: {emission_rate:.0f} kg/hr",
                    icon=folium.Icon(color=color, icon=icon, prefix='fa')
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add measurement control
        from folium.plugins import MeasureControl
        MeasureControl().add_to(m)
        
        # Add fullscreen control
        from folium.plugins import Fullscreen
        Fullscreen().add_to(m)
        
        if save_path:
            m.save(save_path)
            logger.info(f"Interactive map saved to {save_path}")
        
        return m
    
    def create_plotly_dashboard(self, dataset: xr.Dataset, 
                              emission_results: pd.DataFrame) -> go.Figure:
        """Create interactive Plotly dashboard."""
        
        logger.info("Creating Plotly interactive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Enhancement Map', 'Emission Time Series', 
                          'Emitter Locations', 'Emission Distribution'),
            specs=[[{"type": "scattermapbox"}, {"type": "scatter"}],
                   [{"type": "scattermapbox"}, {"type": "histogram"}]]
        )
        
        # 1. Enhancement heatmap (simplified as scatter)
        if 'enhancement' in dataset.data_vars and len(dataset.time) > 0:
            enhancement = dataset.enhancement.isel(time=0)
            lats, lons = np.meshgrid(dataset.lat.values, dataset.lon.values, indexing='ij')
            
            valid_mask = ~np.isnan(enhancement.values)
            if np.any(valid_mask):
                fig.add_trace(
                    go.Scattermapbox(
                        lat=lats[valid_mask].flatten(),
                        lon=lons[valid_mask].flatten(),
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=enhancement.values[valid_mask].flatten(),
                            colorscale='Plasma',
                            colorbar=dict(title="Enhancement (ppb)")
                        ),
                        name='Enhancement',
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # 2. Time series (mock data if no time info)
        if len(emission_results) > 0:
            if 'time' in emission_results.columns:
                time_series = emission_results.groupby('time')['estimated_emission_rate_kg_hr'].sum()
                fig.add_trace(
                    go.Scatter(
                        x=time_series.index,
                        y=time_series.values,
                        mode='lines+markers',
                        name='Total Emissions'
                    ),
                    row=1, col=2
                )
            else:
                # Create mock time series
                dates = pd.date_range(start='2023-01-01', periods=len(emission_results), freq='D')
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=emission_results['estimated_emission_rate_kg_hr'].cumsum(),
                        mode='lines+markers',
                        name='Cumulative Emissions'
                    ),
                    row=1, col=2
                )
        
        # 3. Emitter locations
        if len(emission_results) > 0:
            fig.add_trace(
                go.Scattermapbox(
                    lat=emission_results['center_lat'],
                    lon=emission_results['center_lon'],
                    mode='markers',
                    marker=dict(
                        size=emission_results['estimated_emission_rate_kg_hr'] / 50,
                        color=emission_results['estimated_emission_rate_kg_hr'],
                        colorscale='Reds',
                        colorbar=dict(title="Emission Rate (kg/hr)")
                    ),
                    text=[f"Emitter {i}: {rate:.0f} kg/hr" 
                          for i, rate in enumerate(emission_results['estimated_emission_rate_kg_hr'])],
                    name='Super-Emitters'
                ),
                row=2, col=1
            )
        
        # 4. Emission distribution
        if len(emission_results) > 0:
            fig.add_trace(
                go.Histogram(
                    x=emission_results['estimated_emission_rate_kg_hr'],
                    nbinsx=20,
                    name='Emission Distribution'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Methane Super-Emitter Dashboard",
            height=800,
            showlegend=True
        )
        
        # Update mapbox style
        fig.update_traces(
            selector=dict(type='scattermapbox'),
        )
        
        # Set mapbox style and center
        if len(emission_results) > 0:
            center_lat = emission_results['center_lat'].mean()
            center_lon = emission_results['center_lon'].mean()
        else:
            center_lat = dataset.lat.mean().item()
            center_lon = dataset.lon.mean().item()
        
        fig.update_layout(
            mapbox1=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=6
            ),
            mapbox2=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=6
            )
        )
        
        return fig
    
    def plot_emission_trends(self, tracking_data: pd.DataFrame,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot emission trends for individual emitters."""
        
        logger.info("Creating emission trends plot")
        
        if len(tracking_data) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No tracking data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Top emitters over time
        if 'emitter_id' in tracking_data.columns and 'emission_rate' in tracking_data.columns:
            top_emitters = tracking_data.groupby('emitter_id')['emission_rate'].mean().nlargest(5)
            
            for i, emitter_id in enumerate(top_emitters.index[:4]):
                emitter_data = tracking_data[tracking_data['emitter_id'] == emitter_id]
                if 'timestamp' in emitter_data.columns:
                    axes[i].plot(emitter_data['timestamp'], emitter_data['emission_rate'],
                               marker='o', linewidth=2, label=f'Emitter {emitter_id}')
                    axes[i].set_title(f'Emitter {emitter_id}')
                    axes[i].set_ylabel('Emission Rate (kg/hr)')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trends plot saved to {save_path}")
        
        return fig
    
    def export_visualization_data(self, dataset: xr.Dataset, 
                                emission_results: pd.DataFrame,
                                output_dir: str) -> Dict[str, str]:
        """Export visualization data in multiple formats."""
        
        logger.info(f"Exporting visualization data to {output_dir}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export emission results as CSV
        if len(emission_results) > 0:
            csv_path = output_dir / "super_emitters.csv"
            emission_results.to_csv(csv_path, index=False)
            exported_files['emissions_csv'] = str(csv_path)
        
        # Export enhancement data as NetCDF
        if 'enhancement' in dataset.data_vars:
            nc_path = output_dir / "enhancement_data.nc"
            dataset[['enhancement']].to_netcdf(nc_path)
            exported_files['enhancement_nc'] = str(nc_path)
        
        # Export as GeoJSON for web mapping
        if len(emission_results) > 0:
            geojson_path = output_dir / "super_emitters.geojson"
            
            # Create GeoDataFrame
            from shapely.geometry import Point
            geometry = [Point(lon, lat) for lat, lon in 
                       zip(emission_results['center_lon'], emission_results['center_lat'])]
            gdf = gpd.GeoDataFrame(emission_results, geometry=geometry)
            gdf.to_file(geojson_path, driver='GeoJSON')
            exported_files['emissions_geojson'] = str(geojson_path)
        
        logger.info(f"Exported {len(exported_files)} visualization files")
        return exported_files