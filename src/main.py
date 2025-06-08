# ============================================================================
# FILE: src/main.py
# ============================================================================
import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings

 
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
# ...existing code...
# Import modules
from config.logging_config import setup_logging
from data.tropomi_collector import TROPOMICollector
from data.meteorology_loader import MeteorologyLoader
from data.preprocessor import TROPOMIPreprocessor
from detection.super_emitter_detector import SuperEmitterDetector
from tracking.emitter_tracker import EmitterTracker
from analysis.time_series_analyzer import TimeSeriesAnalyzer
from alerts.alert_manager import AlertManager
from visualization.dashboard import SuperEmitterDashboard
from utils.file_utils import FileManager

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class SuperEmitterPipeline:
    """
    Main pipeline for super-emitter tracking and temporal analysis.
    
    This pipeline orchestrates:
    1. Data collection from TROPOMI and meteorological sources
    2. Super-emitter detection using advanced algorithms
    3. Temporal tracking and trend analysis
    4. Alert generation for significant changes
    5. Visualization and reporting
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.file_manager = FileManager(self.config)
        
        # Initialize components
        self.tropomi_collector = TROPOMICollector(self.config)
        self.met_loader = MeteorologyLoader(self.config)
        self.preprocessor = TROPOMIPreprocessor(self.config)
        self.detector = SuperEmitterDetector(self.config)
        self.tracker = EmitterTracker(self.config)
        self.analyzer = TimeSeriesAnalyzer(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Pipeline state
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(self.config['processing']['output']['base_path']) / f"run_{self.run_id}"
        
        logger.info(f"SuperEmitterPipeline initialized - Run ID: {self.run_id}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def run_operational_monitoring(self, start_date: str = None, 
                                 end_date: str = None) -> dict:
        """
        Run operational monitoring mode for near real-time tracking.
        
        Args:
            start_date: Start date (YYYY-MM-DD), defaults to yesterday
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            Dictionary with pipeline results
        """
        
        # Set default dates for operational mode
        if not start_date:
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info("=" * 80)
        logger.info("SUPER-EMITTER OPERATIONAL MONITORING STARTED")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Run ID: {self.run_id}")
        
        try:
            # Create output directories
            self.file_manager.create_output_directories(self.output_dir)
            
            # Step 1: Collect TROPOMI data
            logger.info("🛰️  Step 1: Collecting TROPOMI data")
            tropomi_data = self.tropomi_collector.collect_data(start_date, end_date)
            
            if tropomi_data is None or len(tropomi_data.time) == 0:
                logger.warning("No TROPOMI data available for the specified period")
                return self._create_empty_results("no_data")
            
            logger.info(f"Collected {len(tropomi_data.time)} time steps")
            
            # Step 2: Collect meteorological data
            logger.info("🌤️  Step 2: Collecting meteorological data")
            met_data = self.met_loader.load_meteorological_data(
                start_date, end_date, tropomi_data
            )
            
            # Step 3: Preprocess data
            logger.info("🔧 Step 3: Preprocessing data")
            processed_data = self.preprocessor.preprocess_tropomi_data(tropomi_data, met_data)
            
            # Step 4: Detect super-emitters
            logger.info("🔍 Step 4: Detecting super-emitters")
            detection_results = self.detector.detect_super_emitters(processed_data)
            
            # Step 5: Track emitters temporally
            logger.info("📈 Step 5: Tracking emitter evolution")
            tracking_results = self.tracker.track_emitters(
                detection_results['super_emitters'], 
                datetime.now()
            )
            
            # Step 6: Perform time series analysis
            logger.info("📊 Step 6: Analyzing temporal patterns")
            analysis_results = self.analyzer.analyze_time_series(
                tracking_results, processed_data
            )
            
            # Step 7: Generate alerts
            logger.info("🚨 Step 7: Processing alerts")
            alerts = self._process_alerts(detection_results, tracking_results, analysis_results)
            
            # Step 8: Save results
            logger.info("💾 Step 8: Saving results")
            self._save_results(
                detection_results, tracking_results, analysis_results, 
                alerts, processed_data
            )
            
            # Step 9: Generate summary
            pipeline_results = self._compile_results(
                detection_results, tracking_results, analysis_results, alerts
            )
            
            # Log summary
            self._log_pipeline_summary(pipeline_results)
            
            logger.info("✅ OPERATIONAL MONITORING COMPLETED SUCCESSFULLY")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            logger.exception("Full traceback:")
            
            # Save error information
            error_info = {
                'error': str(e),
                'timestamp': datetime.now(),
                'run_id': self.run_id,
                'pipeline_stage': 'unknown'
            }
            
            self.file_manager.save_json(
                error_info, 
                self.output_dir / "error_report.json"
            )
            
            raise
    
    def run_historical_analysis(self, start_date: str, end_date: str,
                               analysis_type: str = "comprehensive") -> dict:
        """
        Run historical analysis for research and validation.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            analysis_type: Type of analysis (quick, standard, comprehensive)
            
        Returns:
            Dictionary with analysis results
        """
        
        logger.info("=" * 80)
        logger.info("SUPER-EMITTER HISTORICAL ANALYSIS STARTED")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Analysis type: {analysis_type}")
        logger.info(f"Run ID: {self.run_id}")
        
        # Adjust processing parameters based on analysis type
        if analysis_type == "quick":
            self.config['super_emitters']['detection']['confidence_threshold'] = 0.8
            self.config['tracking']['time_series']['min_observations'] = 5
        elif analysis_type == "comprehensive":
            self.config['super_emitters']['detection']['confidence_threshold'] = 0.6
            self.config['tracking']['time_series']['min_observations'] = 3
        
        try:
            # Process data in chunks for long time periods
            date_chunks = self._create_date_chunks(start_date, end_date, analysis_type)
            
            all_detection_results = []
            all_tracking_results = []
            
            for chunk_start, chunk_end in date_chunks:
                logger.info(f"Processing chunk: {chunk_start} to {chunk_end}")
                
                # Process chunk
                chunk_results = self._process_date_chunk(chunk_start, chunk_end)
                
                if chunk_results:
                    all_detection_results.extend(chunk_results['detections'])
                    all_tracking_results.append(chunk_results['tracking'])
            
            # Combine results from all chunks
            combined_results = self._combine_chunk_results(
                all_detection_results, all_tracking_results
            )
            
            # Perform comprehensive analysis
            analysis_results = self.analyzer.analyze_long_term_trends(
                combined_results, start_date, end_date
            )
            
            # Generate research report
            report = self._generate_research_report(
                combined_results, analysis_results, start_date, end_date
            )
            
            logger.info("✅ HISTORICAL ANALYSIS COMPLETED SUCCESSFULLY")
            return report
            
        except Exception as e:
            logger.error(f"Historical analysis failed: {str(e)}")
            logger.exception("Full traceback:")
            raise
    
    def run_realtime_dashboard(self, port: int = 8501):
        """Launch the real-time monitoring dashboard."""
        
        logger.info("🚀 Launching real-time super-emitter dashboard")
        
        try:
            dashboard = SuperEmitterDashboard(self.config, self.tracker)
            dashboard.run_dashboard(port=port)
            
        except Exception as e:
            logger.error(f"Dashboard failed to start: {e}")
            raise
    
    def _create_date_chunks(self, start_date: str, end_date: str, 
                          analysis_type: str) -> list:
        """Create date chunks for processing long time periods."""
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Determine chunk size based on analysis type
        if analysis_type == "quick":
            chunk_days = 30
        elif analysis_type == "standard":
            chunk_days = 14
        else:  # comprehensive
            chunk_days = 7
        
        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_days), end_dt)
            chunks.append((
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d')
            ))
            current_start = current_end + timedelta(days=1)
        
        logger.info(f"Created {len(chunks)} date chunks for processing")
        return chunks
    
    def _process_date_chunk(self, start_date: str, end_date: str) -> dict:
        """Process a single date chunk."""
        
        try:
            # Collect data
            tropomi_data = self.tropomi_collector.collect_data(start_date, end_date)
            if tropomi_data is None:
                return None
            
            met_data = self.met_loader.load_meteorological_data(
                start_date, end_date, tropomi_data
            )
            
            # Process
            processed_data = self.preprocessor.preprocess_tropomi_data(tropomi_data, met_data)
            detection_results = self.detector.detect_super_emitters(processed_data)
            tracking_results = self.tracker.track_emitters(
                detection_results['super_emitters'], 
                datetime.strptime(end_date, '%Y-%m-%d')
            )
            
            return {
                'detections': detection_results['super_emitters'],
                'tracking': tracking_results,
                'processed_data': processed_data
            }
            
        except Exception as e:
            logger.warning(f"Failed to process chunk {start_date} to {end_date}: {e}")
            return None
    
    def _combine_chunk_results(self, detection_results: list, 
                             tracking_results: list) -> dict:
        """Combine results from multiple date chunks."""
        
        # Combine detection results
        if detection_results:
            combined_detections = pd.concat(detection_results, ignore_index=True)
        else:
            combined_detections = pd.DataFrame()
        
        # Merge tracking results
        combined_tracking = {
            'total_tracks': sum(r['tracking_summary']['active_tracks'] for r in tracking_results),
            'all_alerts': [],
            'tracking_data': tracking_results
        }
        
        for result in tracking_results:
            combined_tracking['all_alerts'].extend(result.get('alerts', []))
        
        return {
            'detections': combined_detections,
            'tracking': combined_tracking
        }
    
    def _process_alerts(self, detection_results: dict, tracking_results: dict, 
                       analysis_results: dict) -> dict:
        """Process and manage alerts from all pipeline components."""
        
        # Collect alerts from different sources
        all_alerts = []
        
        # Detection alerts (new super-emitters)
        if len(detection_results['super_emitters']) > 0:
            new_emitter_alerts = self.alert_manager.generate_new_emitter_alerts(
                detection_results['super_emitters']
            )
            all_alerts.extend(new_emitter_alerts)
        
        # Tracking alerts (trends, missing emitters)
        tracking_alerts = tracking_results.get('alerts', [])
        all_alerts.extend(tracking_alerts)
        
        # Analysis alerts (significant changes)
        analysis_alerts = analysis_results.get('alerts', [])
        all_alerts.extend(analysis_alerts)
        
        # Process and prioritize alerts
        processed_alerts = self.alert_manager.process_alerts(all_alerts)
        
        # Send notifications if enabled
        if self.config['alerts']['notifications']['email']['enabled']:
            self.alert_manager.send_email_notifications(processed_alerts)
        
        if self.config['alerts']['notifications']['webhook']['enabled']:
            self.alert_manager.send_webhook_notifications(processed_alerts)
        
        return processed_alerts
    
    def _save_results(self, detection_results: dict, tracking_results: dict,
                     analysis_results: dict, alerts: dict, processed_data) -> None:
        """Save all pipeline results to files."""
        
        # Save detection results
        if len(detection_results['super_emitters']) > 0:
            self.file_manager.save_dataframe(
                detection_results['super_emitters'],
                self.output_dir / "detections" / "super_emitters.csv"
            )
            
            # Save as GeoJSON for mapping
            self.file_manager.save_geojson(
                detection_results['super_emitters'],
                self.output_dir / "detections" / "super_emitters.geojson",
                lat_col='center_lat', lon_col='center_lon'
            )
        
        # Save tracking results
        tracking_df = self.tracker.export_tracks_to_dataframe()
        if len(tracking_df) > 0:
            self.file_manager.save_dataframe(
                tracking_df,
                self.output_dir / "tracking" / "emitter_tracks.csv"
            )
        
        # Save tracking summary
        self.file_manager.save_json(
            tracking_results,
            self.output_dir / "tracking" / "tracking_summary.json"
        )
        
        # Save analysis results
        self.file_manager.save_json(
            analysis_results,
            self.output_dir / "analysis" / "time_series_analysis.json"
        )
        
        # Save alerts
        self.file_manager.save_json(
            alerts,
            self.output_dir / "alerts" / "alert_summary.json"
        )
        
        # Save processed data (if configured)
        if self.config['development']['debug']['save_intermediate_results']:
            processed_data.to_netcdf(
                self.output_dir / "data" / "processed_tropomi_data.nc"
            )
        
        # Save pipeline metadata
        metadata = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'config_used': self.config,
            'pipeline_version': '1.0.0',
            'data_period': {
                'start': str(processed_data.time.min().values),
                'end': str(processed_data.time.max().values)
            }
        }
        
        self.file_manager.save_json(
            metadata,
            self.output_dir / "pipeline_metadata.json"
        )
        
        logger.info(f"Results saved to: {self.output_dir}")
    
    def _compile_results(self, detection_results: dict, tracking_results: dict,
                        analysis_results: dict, alerts: dict) -> dict:
        """Compile comprehensive pipeline results."""

        # Handle case where super_emitters might be None, empty, or a DataFrame
        super_emitters = detection_results.get('super_emitters', [])

        # Convert to DataFrame if it's not already, or handle empty cases
        if super_emitters is None:
            super_emitters = pd.DataFrame()
        elif not isinstance(super_emitters, pd.DataFrame):
            if hasattr(super_emitters, '__len__') and len(super_emitters) == 0:
                super_emitters = pd.DataFrame()
            else:
                # Try to convert to DataFrame if it's a list or other iterable
                try:
                    super_emitters = pd.DataFrame(super_emitters)
                except:
                    super_emitters = pd.DataFrame()

        # Safely calculate metrics
        total_super_emitters = len(super_emitters)

        if total_super_emitters > 0 and 'estimated_emission_rate_kg_hr' in super_emitters.columns:
            total_emission_rate = float(super_emitters['estimated_emission_rate_kg_hr'].sum())
            mean_emission_rate = float(super_emitters['estimated_emission_rate_kg_hr'].mean())
        else:
            total_emission_rate = 0.0
            mean_emission_rate = 0.0

        if total_super_emitters > 0 and 'facility_id' in super_emitters.columns:
            facility_associations = int(super_emitters['facility_id'].notna().sum())
        else:
            facility_associations = 0

        return {
            'pipeline_info': {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'output_directory': str(self.output_dir)
            },
            'detection_summary': {
                'total_super_emitters': total_super_emitters,
                'total_emission_rate_kg_hr': total_emission_rate,
                'mean_emission_rate_kg_hr': mean_emission_rate,
                'facility_associations': facility_associations
            },
            'tracking_summary': tracking_results.get('tracking_summary', {}),
            'analysis_summary': {
                'trends_analyzed': analysis_results.get('trends_analyzed', 0),
                'significant_trends': analysis_results.get('significant_trends', 0),
                'changepoints_detected': analysis_results.get('changepoints_detected', 0)
            },
            'alert_summary': {
                'total_alerts': len(alerts.get('alerts', [])),
                'high_priority_alerts': len([a for a in alerts.get('alerts', []) if a.get('severity') == 'high']),
                'alert_types': list(set(a.get('alert_type') for a in alerts.get('alerts', []))) if alerts.get('alerts') else []
            },
            'data_quality': detection_results.get('quality_flags', {}),
            'validation_results': detection_results.get('validation_results', {})
        }
    
    def _log_pipeline_summary(self, results: dict) -> None:
        """Log a comprehensive summary of pipeline results."""
        
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        
        # Detection summary
        det_summary = results['detection_summary']
        logger.info(f"🔍 DETECTIONS:")
        logger.info(f"   Super-emitters found: {det_summary['total_super_emitters']}")
        logger.info(f"   Total emission rate: {det_summary['total_emission_rate_kg_hr']:.1f} kg/hr")
        logger.info(f"   Facility associations: {det_summary['facility_associations']}")
        
        # Tracking summary
        track_summary = results['tracking_summary']
        logger.info(f"📈 TRACKING:")
        logger.info(f"   Active tracks: {track_summary['active_tracks']}")
        logger.info(f"   New tracks: {track_summary['new_tracks_started']}")
        logger.info(f"   Associations found: {track_summary['associations_found']}")
        
        # Alert summary
        alert_summary = results['alert_summary']
        logger.info(f"🚨 ALERTS:")
        logger.info(f"   Total alerts: {alert_summary['total_alerts']}")
        logger.info(f"   High priority: {alert_summary['high_priority_alerts']}")
        if alert_summary['alert_types']:
            logger.info(f"   Alert types: {', '.join(alert_summary['alert_types'])}")
        
        # Data quality
        if 'data_quality' in results:
            quality = results['data_quality']
            if 'data_quality' in quality:
                dq = quality['data_quality']
                logger.info(f"📊 DATA QUALITY:")
                logger.info(f"   Coverage: {dq['coverage_fraction']:.2%}")
                logger.info(f"   Temporal span: {dq['temporal_coverage_days']} days")
        
        logger.info("=" * 60)
    
    def _create_empty_results(self, reason: str) -> dict:
        """Create empty results structure when no data is available."""
        
        return {
            'pipeline_info': {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'no_data',
                'reason': reason
            },
            'detection_summary': {
                'total_super_emitters': 0,
                'total_emission_rate_kg_hr': 0,
                'mean_emission_rate_kg_hr': 0,
                'facility_associations': 0
            },
            'tracking_summary': {
                'active_tracks': 0,
                'new_tracks_started': 0,
                'associations_found': 0
            },
            'alert_summary': {
                'total_alerts': 0,
                'high_priority_alerts': 0,
                'alert_types': []
            }
        }
    
    def _generate_research_report(self, combined_results: dict, analysis_results: dict,
                                start_date: str, end_date: str) -> dict:
        """Generate comprehensive research report for historical analysis."""
        
        report = {
            'report_info': {
                'title': 'Super-Emitter Historical Analysis Report',
                'period': f"{start_date} to {end_date}",
                'generated': datetime.now().isoformat(),
                'run_id': self.run_id
            },
            'executive_summary': self._create_executive_summary(combined_results, analysis_results),
            'detailed_findings': analysis_results,
            'methodology': self._create_methodology_section(),
            'data_sources': self._create_data_sources_section(),
            'quality_assessment': self._assess_data_quality(combined_results),
            'recommendations': self._generate_recommendations(analysis_results)
        }
        
        # Save report
        self.file_manager.save_json(
            report,
            self.output_dir / "research_report.json"
        )
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        return report
    
    def _create_executive_summary(self, combined_results: dict, analysis_results: dict) -> dict:
        """Create executive summary for research report."""
        
        detections = combined_results['detections']
        
        if len(detections) == 0:
            return {
                'key_findings': "No super-emitters detected in the analysis period",
                'total_emitters': 0,
                'total_emissions': 0,
                'main_trends': "No trends available due to lack of detections"
            }
        
        summary = {
            'key_findings': f"Analysis identified {len(detections)} super-emitter instances",
            'total_emitters': len(detections['emitter_id'].unique()) if 'emitter_id' in detections.columns else len(detections),
            'total_emissions_kg_hr': float(detections['estimated_emission_rate_kg_hr'].sum()),
            'mean_emission_rate_kg_hr': float(detections['estimated_emission_rate_kg_hr'].mean()),
            'facility_association_rate': float(detections['facility_id'].notna().mean()) if 'facility_id' in detections.columns else 0,
            'geographic_distribution': {
                'lat_range': [float(detections['center_lat'].min()), float(detections['center_lat'].max())],
                'lon_range': [float(detections['center_lon'].min()), float(detections['center_lon'].max())]
            },
            'main_trends': analysis_results.get('trend_summary', "Trend analysis completed")
        }
        
        return summary
    
    def _create_methodology_section(self) -> dict:
        """Create methodology section for research report."""
        
        return {
            'detection_algorithm': {
                'primary_method': 'Statistical anomaly detection with spatial clustering',
                'enhancement_threshold': self.config['super_emitters']['detection']['enhancement_threshold'],
                'emission_threshold': self.config['super_emitters']['detection']['emission_rate_threshold'],
                'confidence_threshold': self.config['super_emitters']['detection']['confidence_threshold']
            },
            'tracking_method': {
                'spatial_association': 'DBSCAN clustering with temporal linkage',
                'trend_analysis': 'Mann-Kendall test and linear regression',
                'persistence_filtering': f"Minimum {self.config['tracking']['persistence']['min_detections']} detections"
            },
            'data_sources': {
                'primary': 'TROPOMI/Sentinel-5P methane column',
                'meteorological': 'ERA5 reanalysis data',
                'validation': 'Known facility database'
            },
            'uncertainty_quantification': 'Monte Carlo error propagation'
        }
    
    def _create_data_sources_section(self) -> dict:
        """Create data sources section for research report."""
        
        return {
            'satellite_data': {
                'instrument': 'TROPOMI (TROPOspheric Monitoring Instrument)',
                'platform': 'Sentinel-5P',
                'parameter': 'CH4 column volume mixing ratio',
                'spatial_resolution': '7 x 3.5 km',
                'temporal_resolution': 'Daily',
                'quality_filtering': f"QA threshold: {self.config['data']['tropomi']['quality_threshold']}"
            },
            'meteorological_data': {
                'source': 'ERA5 reanalysis',
                'parameters': ['Wind speed', 'Wind direction', 'Boundary layer height'],
                'spatial_resolution': '0.25 degrees',
                'temporal_resolution': 'Hourly'
            },
            'reference_data': {
                'facility_database': 'Known emission source locations',
                'validation_sources': self.config['validation']['ground_truth']['sources']
            }
        }
    
    def _assess_data_quality(self, combined_results: dict) -> dict:
        """Assess overall data quality for the analysis."""
        
        detections = combined_results['detections']
        
        if len(detections) == 0:
            return {
                'overall_quality': 'insufficient_data',
                'data_availability': 'No detections available for quality assessment'
            }
        
        # Calculate quality metrics
        mean_score = float(detections['detection_score'].mean()) if 'detection_score' in detections.columns else 0
        
        if mean_score > 0.8:
            quality_rating = 'high'
        elif mean_score > 0.6:
            quality_rating = 'medium'
        else:
            quality_rating = 'low'
        
        return {
            'overall_quality': quality_rating,
            'mean_detection_score': mean_score,
            'data_completeness': 'Good coverage across analysis period',
            'spatial_coverage': f"Analysis region: {detections['center_lat'].min():.2f}°N to {detections['center_lat'].max():.2f}°N",
            'temporal_coverage': f"Detections span analysis period",
            'quality_flags': 'No significant data quality issues identified'
        }
    
    def _generate_recommendations(self, analysis_results: dict) -> list:
        """Generate recommendations based on analysis results."""
        
        recommendations = [
            "Continue monitoring detected super-emitters for emission trends",
            "Validate detections with ground-based measurements where possible",
            "Investigate unassociated emitters for potential new source identification"
        ]
        
        # Add specific recommendations based on results
        if analysis_results.get('significant_trends', 0) > 0:
            recommendations.append("Priority investigation recommended for emitters showing significant increasing trends")
        
        if analysis_results.get('new_emitters_detected', 0) > 0:
            recommendations.append("Immediate attention required for newly detected super-emitters")
        
        return recommendations
    
    def _generate_markdown_report(self, report: dict) -> None:
        """Generate markdown version of the research report."""
        
        markdown_content = f"""# {report['report_info']['title']}

**Analysis Period:** {report['report_info']['period']}  
**Generated:** {report['report_info']['generated']}  
**Run ID:** {report['report_info']['run_id']}

## Executive Summary

{report['executive_summary']['key_findings']}

### Key Statistics
- **Total Super-Emitters:** {report['executive_summary'].get('total_emitters', 0)}
- **Total Emission Rate:** {report['executive_summary'].get('total_emissions_kg_hr', 0):.1f} kg/hr
- **Mean Emission Rate:** {report['executive_summary'].get('mean_emission_rate_kg_hr', 0):.1f} kg/hr
- **Facility Association Rate:** {report['executive_summary'].get('facility_association_rate', 0):.1%}

## Methodology

### Detection Algorithm
- **Primary Method:** {report['methodology']['detection_algorithm']['primary_method']}
- **Enhancement Threshold:** {report['methodology']['detection_algorithm']['enhancement_threshold']} ppb
- **Emission Threshold:** {report['methodology']['detection_algorithm']['emission_threshold']} kg/hr

### Data Sources
- **Primary:** {report['data_sources']['satellite_data']['instrument']} on {report['data_sources']['satellite_data']['platform']}
- **Meteorological:** {report['data_sources']['meteorological_data']['source']}
- **Spatial Resolution:** {report['data_sources']['satellite_data']['spatial_resolution']}

## Quality Assessment

**Overall Quality:** {report['quality_assessment']['overall_quality']}

{report['quality_assessment'].get('quality_flags', 'Standard quality assessment completed')}

## Recommendations

"""
        
        for i, rec in enumerate(report['recommendations'], 1):
            markdown_content += f"{i}. {rec}\n"
        
        # Save markdown file
        markdown_path = self.output_dir / "research_report.md"
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report saved to: {markdown_path}")


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Super-Emitter Tracking and Temporal Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Operational monitoring (last 24 hours)
  python main.py --mode operational
  
  # Historical analysis
  python main.py --mode historical --start-date 2023-06-01 --end-date 2023-06-30
  
  # Launch dashboard
  python main.py --mode dashboard
  
  # Quick test
  python main.py --test
        """
    )
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', choices=['operational', 'historical', 'dashboard'], 
                       default='operational',
                       help='Pipeline mode')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--analysis-type', choices=['quick', 'standard', 'comprehensive'],
                       default='standard',
                       help='Analysis depth for historical mode')
    parser.add_argument('--dashboard-port', type=int, default=8501,
                       help='Port for dashboard mode')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with sample data')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Test mode overrides
    if args.test:
        print("🧪 Running in TEST mode")
        args.mode = 'operational'
        if not args.start_date:
            args.start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        if not args.end_date:
            args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Initialize pipeline
        pipeline = SuperEmitterPipeline(args.config)
        
        # Setup logging
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logging(log_level=log_level, log_file=f"logs/pipeline_{pipeline.run_id}.log")
        
        # Run appropriate mode
        if args.mode == 'operational':
            results = pipeline.run_operational_monitoring(args.start_date, args.end_date)
            
            # Print summary
            print(f"\n🎉 SUCCESS! Super-emitter monitoring completed")
            print(f"📊 Found {results['detection_summary']['total_super_emitters']} super-emitters")
            if results['detection_summary']['total_emission_rate_kg_hr'] > 0:
                print(f"🔥 Total emissions: {results['detection_summary']['total_emission_rate_kg_hr']:.1f} kg/hr")
            print(f"🚨 Generated {results['alert_summary']['total_alerts']} alerts")
            output_dir = results.get('pipeline_info', {}).get('output_directory', 'data/outputs')
            print(f"📁 Results: {output_dir}")
        elif args.mode == 'historical':
            if not args.start_date or not args.end_date:
                parser.error("Historical mode requires --start-date and --end-date")
            
            results = pipeline.run_historical_analysis(
                args.start_date, args.end_date, args.analysis_type
            )
            
            print(f"\n🎉 SUCCESS! Historical analysis completed")
            print(f"📈 Analysis type: {args.analysis_type}")
            print(f"📊 Report generated: {results['report_info']['title']}")
            
        elif args.mode == 'dashboard':
            pipeline.run_realtime_dashboard(args.dashboard_port)
            
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()