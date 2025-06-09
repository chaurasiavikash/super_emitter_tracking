# ============================================================================
# FILE: src/visualization/dashboard.py
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import altair as alt
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)

class SuperEmitterDashboard:
    """
    Interactive Streamlit dashboard for super-emitter monitoring and analysis.
    
    Features:
    - Real-time monitoring of super-emitters
    - Interactive maps with temporal controls
    - Time series analysis and trends
    - Alert management interface
    - Data export capabilities
    - Performance metrics and validation
    """
    
    def __init__(self, config: Dict, tracker=None):
        self.config = config
        self.tracker = tracker
        
        # Dashboard state
        if 'dashboard_data' not in st.session_state:
            st.session_state.dashboard_data = {}
        if 'selected_emitters' not in st.session_state:
            st.session_state.selected_emitters = []
        if 'date_range' not in st.session_state:
            st.session_state.date_range = (
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
    
    def run_dashboard(self, port: int = 8501):
        """Run the Streamlit dashboard."""
        
        st.set_page_config(
            page_title="Super-Emitter Tracking System",
            page_icon="🛰️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        self._inject_custom_css()
        
        # Main dashboard layout
        self._render_dashboard()
    
    def _inject_custom_css(self):
        """Inject custom CSS for dashboard styling."""
        
        st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            border: 1px solid #e1e5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .alert-high {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .alert-medium {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .alert-low {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        .stSelectbox > div > div > select {
            background-color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_dashboard(self):
        """Render the main dashboard interface."""
        
        # Header
        st.title("🛰️ Super-Emitter Tracking System")
        st.markdown("**Real-time monitoring and analysis of methane super-emitters using TROPOMI data**")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ Live Monitoring", 
            "📈 Time Series Analysis", 
            "🚨 Alerts & Notifications",
            "📊 Performance Analytics",
            "⚙️ Data Export"
        ])
        
        with tab1:
            self._render_live_monitoring_tab()
        
        with tab2:
            self._render_time_series_tab()
        
        with tab3:
            self._render_alerts_tab()
        
        with tab4:
            self._render_analytics_tab()
        
        with tab5:
            self._render_export_tab()
    
    def _render_sidebar(self):
        """Render sidebar controls."""
        
        st.sidebar.header("🎛️ Control Panel")
        
        # Data refresh
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            self._refresh_data()
            st.experimental_rerun()
        
        # Date range selection
        st.sidebar.subheader("📅 Time Period")
        
        date_options = st.sidebar.selectbox(
            "Quick Select",
            ["Last 24 hours", "Last 7 days", "Last 30 days", "Custom range"]
        )
        
        if date_options == "Custom range":
            start_date = st.sidebar.date_input("Start Date", 
                                             value=datetime.now() - timedelta(days=30))
            end_date = st.sidebar.date_input("End Date", value=datetime.now())
            st.session_state.date_range = (start_date, end_date)
        else:
            days_map = {"Last 24 hours": 1, "Last 7 days": 7, "Last 30 days": 30}
            days = days_map[date_options]
            st.session_state.date_range = (
                datetime.now() - timedelta(days=days),
                datetime.now()
            )
        
        # Geographic filters
        st.sidebar.subheader("🌍 Geographic Filters")
        
        region_filter = st.sidebar.selectbox(
            "Region",
            ["Global", "North America", "Europe", "Asia", "Custom Bounds"]
        )
        
        if region_filter == "Custom Bounds":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                min_lat = st.number_input("Min Lat", value=30.0, format="%.2f")
                min_lon = st.number_input("Min Lon", value=-120.0, format="%.2f")
            with col2:
                max_lat = st.number_input("Max Lat", value=50.0, format="%.2f")
                max_lon = st.number_input("Max Lon", value=-70.0, format="%.2f")
            
            st.session_state.geographic_bounds = [min_lat, min_lon, max_lat, max_lon]
        
        # Detection thresholds
        st.sidebar.subheader("🎯 Detection Settings")
        
        emission_threshold = st.sidebar.slider(
            "Emission Threshold (kg/hr)",
            min_value=100, max_value=5000, value=1000, step=100
        )
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=1.0, value=0.7, step=0.05
        )
        
        # Data loading status
        st.sidebar.subheader("📊 Data Status")
        
        # Mock data status - in real implementation, get from tracker
        if self.tracker:
            summary = self.tracker.get_tracking_summary()
            st.sidebar.metric("Active Tracks", summary.get('active_tracks', 0))
            st.sidebar.metric("Recent Detections", summary.get('recent_activity_7days', 0))
            st.sidebar.metric("Total Alerts", summary.get('total_alerts', 0))
        else:
            st.sidebar.info("📡 Loading tracking data...")
    
    def _render_live_monitoring_tab(self):
        """Render the live monitoring tab with interactive map."""
        
        col1, col2, col3, col4 = st.columns(4)
        
        
        # Key metrics
        with col1:
            st.metric(
                label="🔥 Active Super-Emitters",
                value="42",
                delta="3 new today"
            )
        
        with col2:
            st.metric(
                label="💨 Total Emission Rate",
                value="15,847 kg/hr",
                delta="+1,250 kg/hr"
            )
        
        with col3:
            st.metric(
                label="🏭 Facility Associations",
                value="38/42",
                delta="90.5%"
            )
        
        with col4:
            st.metric(
                label="🚨 Active Alerts",
                value="7",
                delta="2 high priority"
            )
        
        st.markdown("---")
        
        # Interactive map section
        col_map, col_controls = st.columns([3, 1])
        
        with col_map:
            st.subheader("🗺️ Global Super-Emitter Map")
            
            # Create interactive map
            map_data = self._get_map_data()
            interactive_map = self._create_interactive_map(map_data)
            
            # Display map
            map_result = st_folium(interactive_map, width=800, height=600)
            
            # Handle map interactions
            if map_result['last_object_clicked_tooltip']:
                selected_emitter = map_result['last_object_clicked_tooltip']
                self._display_emitter_details(selected_emitter)
        
        with col_controls:
            st.subheader("🎛️ Map Controls")
            
            # Layer controls
            show_emissions = st.checkbox("Show Emission Rates", value=True)
            show_trends = st.checkbox("Show Trend Arrows", value=False)
            show_facilities = st.checkbox("Show Facilities", value=True)
            show_alerts = st.checkbox("Highlight Alerts", value=True)
            
            # Time animation controls
            st.markdown("**⏱️ Time Animation**")
            play_animation = st.button("▶️ Play")
            
            if play_animation:
                self._animate_time_series()
            
            # Map style
            map_style = st.selectbox(
                "Map Style",
                ["OpenStreetMap", "Satellite", "Terrain", "Dark"]
            )
            
            # Data filters
            st.markdown("**🔍 Data Filters**")
            
            min_emission = st.slider(
                "Min Emission (kg/hr)",
                0, 5000, 1000
            )
            
            facility_types = st.multiselect(
                "Facility Types",
                ["Oil & Gas", "Landfill", "Agriculture", "Unknown"],
                default=["Oil & Gas", "Landfill"]
            )
        
        # Recent detections table
        st.subheader("📋 Recent Detections")
        
        recent_data = self._get_recent_detections()
        
        if not recent_data.empty:
            # Make table interactive
            selected_rows = st.dataframe(
                recent_data,
                column_config={
                    "emitter_id": "Emitter ID",
                    "timestamp": st.column_config.DatetimeColumn("Detection Time"),
                    "emission_rate": st.column_config.NumberColumn(
                        "Emission Rate (kg/hr)",
                        format="%.1f"
                    ),
                    "facility_name": "Facility",
                    "alert_status": st.column_config.SelectboxColumn(
                        "Alert Status",
                        options=["None", "Low", "Medium", "High"]
                    )
                },
                hide_index=True,
                use_container_width=True,
                selection_mode="multi-row"
            )
        else:
            st.info("No recent detections available. Check data connection.")
    
    def _render_time_series_tab(self):
        """Render time series analysis tab."""
        
        st.subheader("📈 Temporal Analysis")
        
        # Emitter selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            emitter_options = self._get_emitter_list()
            selected_emitters = st.multiselect(
                "Select Emitters for Analysis",
                options=emitter_options,
                default=emitter_options[:3] if len(emitter_options) >= 3 else emitter_options
            )
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Emission Trends", "Enhancement Patterns", "Detection Frequency"]
            )
        
        if selected_emitters:
            # Main time series plot
            time_series_data = self._get_time_series_data(selected_emitters)
            
            if not time_series_data.empty:
                fig = self._create_time_series_plot(time_series_data, analysis_type)
                st.plotly_chart(fig, use_container_width=True)
                
                # Trend analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Trend Statistics")
                    trend_stats = self._calculate_trend_statistics(time_series_data)
                    st.dataframe(trend_stats, use_container_width=True)
                
                with col2:
                    st.subheader("🔍 Change Detection")
                    change_points = self._detect_change_points(time_series_data)
                    
                    if not change_points.empty:
                        st.dataframe(change_points, use_container_width=True)
                    else:
                        st.info("No significant change points detected")
                
                # Seasonal decomposition
                st.subheader("🌊 Seasonal Analysis")
                
                if len(time_series_data) > 30:  # Need enough data for seasonal analysis
                    seasonal_fig = self._create_seasonal_decomposition(time_series_data)
                    st.plotly_chart(seasonal_fig, use_container_width=True)
                else:
                    st.info("Insufficient data for seasonal analysis (need >30 observations)")
            else:
                st.warning("No time series data available for selected emitters")
        else:
            st.info("Please select emitters to analyze")
        
        # Comparative analysis
        st.subheader("⚖️ Comparative Analysis")
        
        if len(selected_emitters) > 1:
            comparison_fig = self._create_comparison_plot(selected_emitters)
            st.plotly_chart(comparison_fig, use_container_width=True)
        else:
            st.info("Select multiple emitters for comparison")
    
    def _render_alerts_tab(self):
        """Render alerts and notifications tab."""
        
        st.subheader("🚨 Alert Management")
        
        # Alert summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔴 High Priority", "3", delta="1 new")
        with col2:
            st.metric("🟡 Medium Priority", "12", delta="4 new")
        with col3:
            st.metric("🟢 Low Priority", "7", delta="-2")
        
        # Alert filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.multiselect(
                "Severity",
                ["High", "Medium", "Low"],
                default=["High", "Medium"]
            )
        
        with col2:
            alert_type_filter = st.multiselect(
                "Alert Type",
                ["New Emitter", "Emission Increase", "Missing Emitter", "Data Quality"],
                default=["New Emitter", "Emission Increase"]
            )
        
        with col3:
            time_filter = st.selectbox(
                "Time Period",
                ["Last 24h", "Last 7d", "Last 30d", "All"]
            )
        
        # Active alerts table
        alerts_data = self._get_alerts_data(severity_filter, alert_type_filter, time_filter)
        
        if not alerts_data.empty:
            # Color-code alerts by severity
            def highlight_severity(row):
                if row['severity'] == 'High':
                    return ['background-color: #ffebee'] * len(row)
                elif row['severity'] == 'Medium':
                    return ['background-color: #fff3e0'] * len(row)
                else:
                    return ['background-color: #e8f5e8'] * len(row)
            
            styled_alerts = alerts_data.style.apply(highlight_severity, axis=1)
            st.dataframe(styled_alerts, use_container_width=True)
            
            # Alert actions
            st.subheader("🎯 Alert Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📧 Send Email Report"):
                    self._send_email_report()
                    st.success("Email report sent!")
            
            with col2:
                if st.button("✅ Mark as Resolved"):
                    st.success("Selected alerts marked as resolved")
            
            with col3:
                if st.button("🔄 Refresh Alerts"):
                    st.experimental_rerun()
        else:
            st.info("No alerts match current filters")
        
        # Alert timeline
        st.subheader("📅 Alert Timeline")
        timeline_fig = self._create_alert_timeline()
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Notification settings
        with st.expander("⚙️ Notification Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.checkbox("Email Notifications", value=True)
                st.checkbox("Webhook Notifications", value=False)
                st.checkbox("Dashboard Alerts", value=True)
            
            with col2:
                email_frequency = st.selectbox(
                    "Email Frequency",
                    ["Immediate", "Hourly", "Daily", "Weekly"]
                )
                
                alert_threshold = st.slider(
                    "Alert Threshold",
                    min_value=0.1, max_value=1.0, value=0.7
                )
    
    def _render_analytics_tab(self):
        """Render performance analytics tab."""
        
        st.subheader("📊 System Performance Analytics")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Detection Rate", "95.3%", delta="2.1%")
        with col2:
            st.metric("False Positive Rate", "8.7%", delta="-1.2%")
        with col3:
            st.metric("Processing Time", "2.3 min", delta="-0.5 min")
        with col4:
            st.metric("Data Coverage", "92.1%", delta="1.8%")
        
        # Performance over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Detection Performance")
            performance_fig = self._create_performance_plot()
            st.plotly_chart(performance_fig, use_container_width=True)
        
        with col2:
            st.subheader("⏱️ Processing Statistics")
            processing_fig = self._create_processing_stats_plot()
            st.plotly_chart(processing_fig, use_container_width=True)
        
        # Validation results
        st.subheader("✅ Validation Results")
        
        validation_data = self._get_validation_data()
        
        if not validation_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion matrix
                confusion_fig = self._create_confusion_matrix()
                st.plotly_chart(confusion_fig, use_container_width=True)
            
            with col2:
                # ROC curve
                roc_fig = self._create_roc_curve()
                st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.info("No validation data available")
        
        # Error analysis
        with st.expander("🔍 Error Analysis"):
            error_analysis = self._get_error_analysis()
            st.dataframe(error_analysis, use_container_width=True)
    
    def _render_export_tab(self):
        """Render data export tab."""
        
        st.subheader("📥 Data Export")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗂️ Export Data")
            
            export_type = st.selectbox(
                "Data Type",
                ["Super-Emitter Detections", "Time Series Data", "Alert History", "Validation Results"]
            )
            
            export_format = st.selectbox(
                "Format",
                ["CSV", "JSON", "GeoJSON", "NetCDF", "Excel"]
            )
            
            date_range = st.date_input(
                "Date Range",
                value=[datetime.now() - timedelta(days=30), datetime.now()],
                max_value=datetime.now()
            )
            
            if st.button("📥 Export Data"):
                export_data = self._prepare_export_data(export_type, export_format, date_range)
                
                if export_data:
                    st.download_button(
                        label=f"Download {export_type} ({export_format})",
                        data=export_data,
                        file_name=f"{export_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.{export_format.lower()}",
                        mime=self._get_mime_type(export_format)
                    )
                else:
                    st.error("Failed to prepare export data")
        
        with col2:
            st.subheader("📊 Generate Reports")
            
            report_type = st.selectbox(
                "Report Type",
                ["Executive Summary", "Technical Report", "Validation Report", "Alert Summary"]
            )
            
            report_period = st.selectbox(
                "Period",
                ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"]
            )
            
            include_plots = st.checkbox("Include Visualizations", value=True)
            include_raw_data = st.checkbox("Include Raw Data", value=False)
            
            if st.button("📄 Generate Report"):
                report_data = self._generate_report(
                    report_type, report_period, include_plots, include_raw_data
                )
                
                st.download_button(
                    label=f"Download {report_type}",
                    data=report_data,
                    file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
        
        # Data summary
        st.subheader("📈 Data Summary")
        
        summary_stats = self._get_data_summary()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", summary_stats.get('total_records', 0))
            st.metric("Date Range", f"{summary_stats.get('date_range', 'N/A')} days")
        
        with col2:
            st.metric("Unique Emitters", summary_stats.get('unique_emitters', 0))
            st.metric("Data Completeness", f"{summary_stats.get('completeness', 0):.1%}")
        
        with col3:
            st.metric("File Size", f"{summary_stats.get('file_size_mb', 0):.1f} MB")
            st.metric("Last Updated", summary_stats.get('last_updated', 'Unknown'))
    
    # Helper methods for data retrieval and processing
    
    def _refresh_data(self):
        """Refresh dashboard data."""
        if self.tracker:
            st.session_state.dashboard_data = {
                'last_refresh': datetime.now(),
                'tracking_summary': self.tracker.get_tracking_summary()
            }
        else:
            # Mock data refresh
            st.session_state.dashboard_data = {
                'last_refresh': datetime.now(),
                'mock_data': True
            }
    def _load_pipeline_data(self):
        """Load the most recent pipeline results."""
        import glob
        import os
        
        # Find the most recent run directory - FIXED PATH
        output_dirs = glob.glob("./data/outputs/run_*")  # Changed from ../../data/outputs/run_*
        if not output_dirs:
            st.warning("No pipeline runs found in ./data/outputs/")
            return None
        
        latest_run = max(output_dirs, key=os.path.getctime)
        st.info(f"Loading data from: {latest_run}")  # Add this to see what it finds
        
        try:
            # Load super-emitters data
            emitters_file = os.path.join(latest_run, "detections", "super_emitters.csv")
            st.info(f"Looking for file: {emitters_file}")  # Add this debug info
            
            if os.path.exists(emitters_file):
                data = pd.read_csv(emitters_file)
                st.success(f"Loaded {len(data)} super-emitters!")  # Add success message
                return data
            else:
                st.error(f"File not found: {emitters_file}")
        except Exception as e:
            st.error(f"Error loading data: {e}")
        
        return None
    def _get_map_data(self) -> pd.DataFrame:
        """Get data for the interactive map."""
        real_data = self._load_pipeline_data()

        if real_data is not None and not real_data.empty:
            # Map your real columns to dashboard expected columns
            mapped_data = pd.DataFrame({
                'emitter_id': real_data['emitter_id'],
                'lat': real_data['center_lat'],
                'lon': real_data['center_lon'],
                'emission_rate': real_data['estimated_emission_rate_kg_hr'],
                'facility_type': real_data.get('facility_type', 'Unknown'),
                'alert_status': 'High',  # Since these are super-emitters
                'last_detection': real_data.get('first_detected', pd.Timestamp.now())
            })
            return mapped_data
        else:
            # Fall back to mock data if no real data found
            st.warning("No pipeline data found. Showing mock data.")
            # ... keep your existing mock data code below
            np.random.seed(42)
            n_emitters = 25

            data = {
                'emitter_id': [f'SE_{i:04d}' for i in range(n_emitters)],
                'lat': np.random.uniform(30, 50, n_emitters),
                'lon': np.random.uniform(-120, -70, n_emitters),
                'emission_rate': np.random.lognormal(7, 0.5, n_emitters),
                'facility_type': np.random.choice(['Oil & Gas', 'Landfill', 'Agriculture'], n_emitters),
                'alert_status': np.random.choice(['None', 'Low', 'Medium', 'High'], n_emitters, p=[0.6, 0.2, 0.15, 0.05]),
                'last_detection': [datetime.now() - timedelta(hours=np.random.randint(1, 72)) for _ in range(n_emitters)]
            }

            return pd.DataFrame(data) 
         
    
    def _create_interactive_map(self, data: pd.DataFrame):
        """Create interactive Folium map."""
        # Center map on data
        center_lat = data['lat'].mean()
        center_lon = data['lon'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        
        # Color mapping for alert status
        color_map = {'None': 'green', 'Low': 'blue', 'Medium': 'orange', 'High': 'red'}
        
        for _, row in data.iterrows():
            # Size marker by emission rate
            radius = max(5, min(20, row['emission_rate'] / 100))
            color = color_map.get(row['alert_status'], 'gray')
            
            popup_text = f"""
            <b>Emitter:</b> {row['emitter_id']}<br>
            <b>Emission Rate:</b> {row['emission_rate']:.0f} kg/hr<br>
            <b>Facility Type:</b> {row['facility_type']}<br>
            <b>Alert Status:</b> {row['alert_status']}<br>
            <b>Last Detection:</b> {row['last_detection'].strftime('%Y-%m-%d %H:%M')}
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                popup=popup_text,
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        return m
    
    def _get_recent_detections(self) -> pd.DataFrame:
        """Get recent detection data."""
        # Mock data
        n_detections = 10
        
        data = {
            'emitter_id': [f'SE_{i:04d}' for i in range(n_detections)],
            'timestamp': [datetime.now() - timedelta(hours=np.random.randint(1, 24)) for _ in range(n_detections)],
            'emission_rate': np.random.lognormal(7, 0.5, n_detections),
            'facility_name': [f'Facility_{i}' for i in range(n_detections)],
            'alert_status': np.random.choice(['None', 'Low', 'Medium', 'High'], n_detections),
            'confidence': np.random.uniform(0.6, 0.95, n_detections)
        }
        
        return pd.DataFrame(data).sort_values('timestamp', ascending=False)
    
    def _get_emitter_list(self) -> List[str]:
        """Get list of available emitters."""
        return [f'SE_{i:04d}' for i in range(50)]
    
    def _get_time_series_data(self, emitters: List[str]) -> pd.DataFrame:
        """Get time series data for selected emitters."""
        # Mock time series data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        data = []
        for emitter in emitters:
            for date in dates:
                if np.random.random() > 0.3:  # Some missing data
                    data.append({
                        'emitter_id': emitter,
                        'timestamp': date,
                        'emission_rate': np.random.lognormal(7, 0.3) + np.sin(len(data) * 0.1) * 200,
                        'enhancement': np.random.normal(50, 15),
                        'detection_score': np.random.uniform(0.6, 0.95)
                    })
        
        return pd.DataFrame(data)
    
    def _create_time_series_plot(self, data: pd.DataFrame, analysis_type: str):
        """Create time series plot."""
        y_column = {
            'Emission Trends': 'emission_rate',
            'Enhancement Patterns': 'enhancement',
            'Detection Frequency': 'detection_score'
        }[analysis_type]
        
        fig = px.line(
            data, 
            x='timestamp', 
            y=y_column, 
            color='emitter_id',
            title=f'{analysis_type} Over Time',
            labels={y_column: analysis_type.split()[0]}
        )
        
        fig.update_layout(height=500)
        return fig
    
    def _calculate_trend_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend statistics."""
        # Mock trend statistics
        emitters = data['emitter_id'].unique()
        
        stats = []
        for emitter in emitters:
            emitter_data = data[data['emitter_id'] == emitter]
            
            stats.append({
                'Emitter': emitter,
                'Trend': np.random.choice(['Increasing', 'Decreasing', 'Stable']),
                'Slope': np.random.normal(0, 50),
                'R²': np.random.uniform(0.3, 0.9),
                'P-value': np.random.uniform(0.001, 0.1),
                'Significance': np.random.choice(['Significant', 'Not Significant'])
            })
        
        return pd.DataFrame(stats)
    
    def _detect_change_points(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect change points in time series."""
        # Mock change points
        change_points = []
        
        for emitter in data['emitter_id'].unique()[:3]:  # Limit to avoid clutter
            if np.random.random() > 0.5:
                change_points.append({
                    'Emitter': emitter,
                    'Change Date': datetime.now() - timedelta(days=np.random.randint(5, 25)),
                    'Change Type': np.random.choice(['Increase', 'Decrease']),
                    'Magnitude': np.random.uniform(100, 500),
                    'Confidence': np.random.uniform(0.7, 0.95)
                })
        
        return pd.DataFrame(change_points)
    
    def _create_seasonal_decomposition(self, data: pd.DataFrame):
        """Create seasonal decomposition plot."""
        # Mock seasonal decomposition
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.05
        )
        
        # Add mock decomposition components
        for i, component in enumerate(['original', 'trend', 'seasonal', 'residual']):
            y_values = np.random.normal(1000, 100, len(dates))
            
            fig.add_trace(
                go.Scatter(x=dates, y=y_values, name=component.title()),
                row=i+1, col=1
            )
        
        fig.update_layout(height=600, showlegend=False)
        return fig
    
    def _create_comparison_plot(self, emitters: List[str]):
        """Create comparison plot for multiple emitters."""
        # Mock comparison data
        categories = ['Emission Rate', 'Detection Frequency', 'Trend Strength', 'Data Quality']
        
        fig = go.Figure()
        
        for emitter in emitters[:5]:  # Limit to 5 emitters
            values = np.random.uniform(0.3, 1.0, len(categories))
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=emitter
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Multi-Emitter Comparison"
        )
        
        return fig
    
    def _get_alerts_data(self, severity_filter: List[str], 
                        alert_type_filter: List[str], time_filter: str) -> pd.DataFrame:
        """Get filtered alerts data."""
        # Mock alerts data
        n_alerts = 20
        
        data = {
            'alert_id': [f'ALT_{i:04d}' for i in range(n_alerts)],
            'timestamp': [datetime.now() - timedelta(hours=np.random.randint(1, 168)) for _ in range(n_alerts)],
            'severity': np.random.choice(['High', 'Medium', 'Low'], n_alerts),
            'alert_type': np.random.choice(['New Emitter', 'Emission Increase', 'Missing Emitter', 'Data Quality'], n_alerts),
            'emitter_id': [f'SE_{i:04d}' for i in np.random.randint(0, 100, n_alerts)],
            'message': ['Sample alert message' for _ in range(n_alerts)],
            'status': np.random.choice(['Active', 'Acknowledged', 'Resolved'], n_alerts)
        }
        
        df = pd.DataFrame(data)
        
        # Apply filters
        if severity_filter:
            df = df[df['severity'].isin(severity_filter)]
        if alert_type_filter:
            df = df[df['alert_type'].isin(alert_type_filter)]
        
        return df.sort_values('timestamp', ascending=False)
    
    def _create_alert_timeline(self):
        """Create alert timeline visualization."""
        # Mock timeline data
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
        
        alert_counts = {
            'High': np.random.poisson(0.5, len(dates)),
            'Medium': np.random.poisson(1.2, len(dates)),
            'Low': np.random.poisson(0.8, len(dates))
        }
        
        fig = go.Figure()
        
        for severity, counts in alert_counts.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=counts,
                mode='lines+markers',
                name=f'{severity} Priority',
                stackgroup='one'
            ))
        
        fig.update_layout(
            title='Alert Frequency Over Time',
            xaxis_title='Time',
            yaxis_title='Number of Alerts',
            height=400
        )
        
        return fig
    
    # Additional helper methods for other dashboard functions...
    
    def _send_email_report(self):
        """Send email report (mock function)."""
        pass
    
    def _create_performance_plot(self):
        """Create performance metrics plot."""
        # Mock performance data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=np.random.uniform(90, 98, len(dates)),
            mode='lines+markers',
            name='Detection Rate (%)',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=np.random.uniform(5, 15, len(dates)),
            mode='lines+markers',
            name='False Positive Rate (%)',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Detection Performance Over Time',
            xaxis_title='Date',
            yaxis=dict(title='Detection Rate (%)', side='left'),
            yaxis2=dict(title='False Positive Rate (%)', side='right', overlaying='y'),
            height=400
        )
        
        return fig
    
    def _create_processing_stats_plot(self):
        """Create processing statistics plot."""
        # Mock processing stats
        categories = ['Data Collection', 'Preprocessing', 'Detection', 'Tracking', 'Analysis']
        times = np.random.uniform(0.5, 5.0, len(categories))
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=times, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title='Processing Time by Stage',
            xaxis_title='Pipeline Stage',
            yaxis_title='Time (minutes)',
            height=400
        )
        
        return fig
    
    def _get_validation_data(self) -> pd.DataFrame:
        """Get validation data."""
        # Mock validation data
        return pd.DataFrame({
            'metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
            'value': np.random.uniform(0.8, 0.95, 4)
        })
    
    def _create_confusion_matrix(self):
        """Create confusion matrix visualization."""
        # Mock confusion matrix
        matrix = np.random.randint(5, 50, (2, 2))
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            text=matrix,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        
        fig# ============================================================================
# FILE: src/visualization/dashboard.py
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import altair as alt
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)

class SuperEmitterDashboard:
    """
    Interactive Streamlit dashboard for super-emitter monitoring and analysis.
    
    Features:
    - Real-time monitoring of super-emitters
    - Interactive maps with temporal controls
    - Time series analysis and trends
    - Alert management interface
    - Data export capabilities
    - Performance metrics and validation
    """
    
    def __init__(self, config: Dict, tracker=None):
        self.config = config
        self.tracker = tracker
        
        # Dashboard state
        if 'dashboard_data' not in st.session_state:
            st.session_state.dashboard_data = {}
        if 'selected_emitters' not in st.session_state:
            st.session_state.selected_emitters = []
        if 'date_range' not in st.session_state:
            st.session_state.date_range = (
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
    
    def run_dashboard(self, port: int = 8501):
        """Run the Streamlit dashboard."""
        
        st.set_page_config(
            page_title="Super-Emitter Tracking System",
            page_icon="🛰️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        self._inject_custom_css()
        
        # Main dashboard layout
        self._render_dashboard()
    
    def _inject_custom_css(self):
        """Inject custom CSS for dashboard styling."""
        
        st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            border: 1px solid #e1e5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .alert-high {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .alert-medium {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .alert-low {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        .stSelectbox > div > div > select {
            background-color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_dashboard(self):
        """Render the main dashboard interface."""
        
        # Header
        st.title("🛰️ Super-Emitter Tracking System")
        st.markdown("**Real-time monitoring and analysis of methane super-emitters using TROPOMI data**")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ Live Monitoring", 
            "📈 Time Series Analysis", 
            "🚨 Alerts & Notifications",
            "📊 Performance Analytics",
            "⚙️ Data Export"
        ])
        
        with tab1:
            self._render_live_monitoring_tab()
        
        with tab2:
            self._render_time_series_tab()
        
        with tab3:
            self._render_alerts_tab()
        
        with tab4:
            self._render_analytics_tab()
        
        with tab5:
            self._render_export_tab()
    
    def _render_sidebar(self):
        """Render sidebar controls."""
        
        st.sidebar.header("🎛️ Control Panel")
        
        # Data refresh
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            self._refresh_data()
            st.experimental_rerun()
        
        # Date range selection
        st.sidebar.subheader("📅 Time Period")
        
        date_options = st.sidebar.selectbox(
            "Quick Select",
            ["Last 24 hours", "Last 7 days", "Last 30 days", "Custom range"]
        )
        
        if date_options == "Custom range":
            start_date = st.sidebar.date_input("Start Date", 
                                             value=datetime.now() - timedelta(days=30))
            end_date = st.sidebar.date_input("End Date", value=datetime.now())
            st.session_state.date_range = (start_date, end_date)
        else:
            days_map = {"Last 24 hours": 1, "Last 7 days": 7, "Last 30 days": 30}
            days = days_map[date_options]
            st.session_state.date_range = (
                datetime.now() - timedelta(days=days),
                datetime.now()
            )
        
        # Geographic filters
        st.sidebar.subheader("🌍 Geographic Filters")
        
        region_filter = st.sidebar.selectbox(
            "Region",
            ["Global", "North America", "Europe", "Asia", "Custom Bounds"]
        )
        
        if region_filter == "Custom Bounds":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                min_lat = st.number_input("Min Lat", value=30.0, format="%.2f")
                min_lon = st.number_input("Min Lon", value=-120.0, format="%.2f")
            with col2:
                max_lat = st.number_input("Max Lat", value=50.0, format="%.2f")
                max_lon = st.number_input("Max Lon", value=-70.0, format="%.2f")
            
            st.session_state.geographic_bounds = [min_lat, min_lon, max_lat, max_lon]
        
        # Detection thresholds
        st.sidebar.subheader("🎯 Detection Settings")
        
        emission_threshold = st.sidebar.slider(
            "Emission Threshold (kg/hr)",
            min_value=100, max_value=5000, value=1000, step=100
        )
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=1.0, value=0.7, step=0.05
        )
        
        # Data loading status
        st.sidebar.subheader("📊 Data Status")
        
        # Mock data status - in real implementation, get from tracker
        if self.tracker:
            summary = self.tracker.get_tracking_summary()
            st.sidebar.metric("Active Tracks", summary.get('active_tracks', 0))
            st.sidebar.metric("Recent Detections", summary.get('recent_activity_7days', 0))
            st.sidebar.metric("Total Alerts", summary.get('total_alerts', 0))
        else:
            st.sidebar.info("📡 Loading tracking data...")
    
    def _render_live_monitoring_tab(self):
        """Render the live monitoring tab with interactive map."""

        col1, col2, col3, col4 = st.columns(4)

        # Key metrics
        with col1:
            st.metric(
                label="🔥 Active Super-Emitters",
                value="42",
                delta="3 new today"
            )

        with col2:
            st.metric(
                label="💨 Total Emission Rate",
                value="15,847 kg/hr",
                delta="+1,250 kg/hr"
            )

        with col3:
            st.metric(
                label="🏭 Facility Associations",
                value="38/42",
                delta="90.5%"
            )

        with col4:
            st.metric(
                label="🚨 Active Alerts",
                value="7",
                delta="2 high priority"
            )

        st.markdown("---")

        # Interactive map section
        col_map, col_controls = st.columns([3, 1])

        with col_map:
            st.subheader("🗺️ Global Super-Emitter Map")

            # Create interactive map
            map_data = self._get_map_data()
            interactive_map = self._create_interactive_map(map_data)

            # Display map
            map_result = st_folium(interactive_map, width=800, height=600)

            # Handle map interactions
            if map_result['last_object_clicked_tooltip']:
                selected_emitter = map_result['last_object_clicked_tooltip']
                self._display_emitter_details(selected_emitter)

        with col_controls:
            st.subheader("🎛️ Map Controls")

            # Layer controls
            show_emissions = st.checkbox("Show Emission Rates", value=True)
            show_trends = st.checkbox("Show Trend Arrows", value=False)
            show_facilities = st.checkbox("Show Facilities", value=True)
            show_alerts = st.checkbox("Highlight Alerts", value=True)

            # Time animation controls
            st.markdown("**⏱️ Time Animation**")
            play_animation = st.button("▶️ Play")

            if play_animation:
                self._animate_time_series()

            # Map style
            map_style = st.selectbox(
                "Map Style",
                ["OpenStreetMap", "Satellite", "Terrain", "Dark"]
            )

            # Data filters
            st.markdown("**🔍 Data Filters**")

            min_emission = st.slider(
                "Min Emission (kg/hr)",
                0, 5000, 1000
            )

            facility_types = st.multiselect(
                "Facility Types",
                ["Oil & Gas", "Landfill", "Agriculture", "Unknown"],
                default=["Oil & Gas", "Landfill"]
            )

        # Recent detections table
        st.subheader("📋 Recent Detections")

        recent_data = self._get_recent_detections()

        if not recent_data.empty:
            # Make table interactive
            selected_rows = st.dataframe(
                recent_data,
                column_config={
                    "emitter_id": "Emitter ID",
                    "timestamp": st.column_config.DatetimeColumn("Detection Time"),
                    "emission_rate": st.column_config.NumberColumn(
                        "Emission Rate (kg/hr)",
                        format="%.1f"
                    ),
                    "facility_name": "Facility",
                    "alert_status": st.column_config.SelectboxColumn(
                        "Alert Status",
                        options=["None", "Low", "Medium", "High"]
                    )
                },
                hide_index=True,
                use_container_width=True,
                selection_mode="multi-row"
            )
        else:
            st.info("No recent detections available. Check data connection.")

    def _display_emitter_details(self, selected_emitter):
        """Display details for selected emitter."""
        st.subheader("📍 Emitter Details")

        with st.expander("Detailed Information", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Emitter ID:** {selected_emitter.get('emitter_id', 'Unknown')}")
                st.write(f"**Location:** {selected_emitter.get('lat', 0):.3f}°N, {selected_emitter.get('lon', 0):.3f}°W")
                st.write(f"**Emission Rate:** {selected_emitter.get('emission_rate', 0):.1f} kg/hr")

            with col2:
                st.write(f"**Facility Type:** {selected_emitter.get('facility_type', 'Unknown')}")
                st.write(f"**Alert Status:** {selected_emitter.get('alert_status', 'None')}")
                st.write(f"**Last Detection:** {selected_emitter.get('last_detection', 'Unknown')}")

    def _animate_time_series(self):
        """Animate time series data on map."""
        st.info("Time animation feature coming soon!")

    def _get_mime_type(self, format_type: str) -> str:
        """Get MIME type for file format."""
        mime_types = {
            'csv': 'text/csv',
            'json': 'application/json',
            'geojson': 'application/geo+json',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'pdf': 'application/pdf'
        }
        return mime_types.get(format_type.lower(), 'application/octet-stream')

    def _prepare_export_data(self, export_type: str, export_format: str, date_range) -> Optional[bytes]:
        """Prepare data for export."""
        # Mock implementation - replace with actual data preparation
        if export_type == "Super-Emitter Detections":
            data = self._get_recent_detections()
            if export_format.lower() == 'csv':
                return data.to_csv(index=False).encode('utf-8')
            elif export_format.lower() == 'json':
                return data.to_json(orient='records', indent=2).encode('utf-8')

        return None

    def _generate_report(self, report_type: str, report_period: str, 
                        include_plots: bool, include_raw_data: bool) -> bytes:
        """Generate PDF report."""
        # Mock implementation - would use reportlab or similar for real PDF generation
        report_content = f"""
        {report_type}
        Period: {report_period}
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Summary:
        - Plots included: {include_plots}
        - Raw data included: {include_raw_data}

        This is a mock report. In production, this would generate a proper PDF.
        """
        return report_content.encode('utf-8')

    def _get_data_summary(self) -> Dict:
        """Get data summary statistics."""
        return {
            'total_records': 1547,
            'unique_emitters': 42,
            'date_range': 30,
            'completeness': 0.923,
            'file_size_mb': 15.7,
            'last_updated': '2 hours ago'
        }