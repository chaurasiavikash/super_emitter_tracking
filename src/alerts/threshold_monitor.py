# ============================================================================
# FILE: src/alerts/threshold_monitor.py
# ============================================================================
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThresholdType(Enum):
    """Types of threshold monitoring."""
    EMISSION_RATE = "emission_rate"
    ENHANCEMENT = "enhancement"
    DETECTION_SCORE = "detection_score"
    TREND_CHANGE = "trend_change"
    SPATIAL_DENSITY = "spatial_density"
    DATA_QUALITY = "data_quality"

@dataclass
class ThresholdConfig:
    """Configuration for a monitoring threshold."""
    threshold_type: ThresholdType
    threshold_value: float
    comparison_operator: str  # '>', '<', '>=', '<=', '==', '!='
    alert_severity: AlertSeverity
    time_window_hours: Optional[int] = None
    min_detections: int = 1
    enabled: bool = True
    description: str = ""

@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    timestamp: datetime
    threshold_type: ThresholdType
    severity: AlertSeverity
    emitter_id: Optional[str]
    value: float
    threshold_value: float
    message: str
    metadata: Dict
    acknowledged: bool = False
    resolved: bool = False

class ThresholdMonitor:
    """
    Monitor various thresholds and generate alerts for super-emitter tracking system.
    
    Features:
    - Configurable threshold monitoring
    - Multiple alert severity levels
    - Time-based and count-based thresholds
    - Alert deduplication and rate limiting
    - Historical alert tracking
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_config = config.get('alerts', {})
        
        # Alert storage
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Threshold configurations
        self.thresholds: List[ThresholdConfig] = []
        
        # Alert rate limiting
        self.rate_limit_window = timedelta(minutes=30)  # Minimum time between similar alerts
        
        self._initialize_default_thresholds()
        
        logger.info("ThresholdMonitor initialized")
    
    def _initialize_default_thresholds(self):
        """Initialize default threshold configurations."""
        
        # Get threshold values from config
        thresholds = self.alert_config.get('thresholds', {})
        
        default_thresholds = [
            # Emission rate thresholds
            ThresholdConfig(
                threshold_type=ThresholdType.EMISSION_RATE,
                threshold_value=thresholds.get('high_emission_rate', 2000.0),
                comparison_operator='>',
                alert_severity=AlertSeverity.HIGH,
                description="High emission rate detected"
            ),
            ThresholdConfig(
                threshold_type=ThresholdType.EMISSION_RATE,
                threshold_value=thresholds.get('critical_emission_rate', 5000.0),
                comparison_operator='>',
                alert_severity=AlertSeverity.CRITICAL,
                description="Critical emission rate detected"
            ),
            
            # Enhancement thresholds
            ThresholdConfig(
                threshold_type=ThresholdType.ENHANCEMENT,
                threshold_value=thresholds.get('high_enhancement', 100.0),
                comparison_operator='>',
                alert_severity=AlertSeverity.MEDIUM,
                description="High methane enhancement detected"
            ),
            
            # Detection score thresholds
            ThresholdConfig(
                threshold_type=ThresholdType.DETECTION_SCORE,
                threshold_value=thresholds.get('high_confidence', 0.9),
                comparison_operator='>',
                alert_severity=AlertSeverity.MEDIUM,
                description="High confidence detection"
            ),
            
            # Data quality thresholds
            ThresholdConfig(
                threshold_type=ThresholdType.DATA_QUALITY,
                threshold_value=thresholds.get('low_data_quality', 0.5),
                comparison_operator='<',
                alert_severity=AlertSeverity.MEDIUM,
                description="Low data quality detected"
            ),
            
            # Spatial density threshold
            ThresholdConfig(
                threshold_type=ThresholdType.SPATIAL_DENSITY,
                threshold_value=thresholds.get('high_spatial_density', 5),
                comparison_operator='>',
                alert_severity=AlertSeverity.MEDIUM,
                time_window_hours=24,
                description="High spatial density of emitters"
            )
        ]
        
        self.thresholds.extend(default_thresholds)
    
    def add_threshold(self, threshold_config: ThresholdConfig):
        """Add a new threshold configuration."""
        self.thresholds.append(threshold_config)
        logger.info(f"Added threshold: {threshold_config.description}")
    
    def remove_threshold(self, threshold_type: ThresholdType, threshold_value: float):
        """Remove a threshold configuration."""
        self.thresholds = [
            t for t in self.thresholds 
            if not (t.threshold_type == threshold_type and t.threshold_value == threshold_value)
        ]
        logger.info(f"Removed threshold: {threshold_type.value} = {threshold_value}")
    
    def monitor_detections(self, detections: pd.DataFrame) -> List[Alert]:
        """
        Monitor detection data against configured thresholds.
        
        Args:
            detections: DataFrame with super-emitter detections
            
        Returns:
            List of generated alerts
        """
        logger.info(f"Monitoring {len(detections)} detections against {len(self.thresholds)} thresholds")
        
        if len(detections) == 0:
            return []
        
        new_alerts = []
        
        for threshold in self.thresholds:
            if not threshold.enabled:
                continue
            
            try:
                alerts = self._check_threshold(detections, threshold)
                new_alerts.extend(alerts)
            except Exception as e:
                logger.error(f"Error checking threshold {threshold.threshold_type.value}: {e}")
        
        # Apply rate limiting
        filtered_alerts = self._apply_rate_limiting(new_alerts)
        
        # Store alerts
        self.active_alerts.extend(filtered_alerts)
        self.alert_history.extend(filtered_alerts)
        
        logger.info(f"Generated {len(filtered_alerts)} new alerts")
        return filtered_alerts
    
    def _check_threshold(self, detections: pd.DataFrame, threshold: ThresholdConfig) -> List[Alert]:
        """Check a specific threshold against detection data."""
        
        alerts = []
        
        if threshold.threshold_type == ThresholdType.EMISSION_RATE:
            alerts = self._check_emission_rate_threshold(detections, threshold)
            
        elif threshold.threshold_type == ThresholdType.ENHANCEMENT:
            alerts = self._check_enhancement_threshold(detections, threshold)
            
        elif threshold.threshold_type == ThresholdType.DETECTION_SCORE:
            alerts = self._check_detection_score_threshold(detections, threshold)
            
        elif threshold.threshold_type == ThresholdType.SPATIAL_DENSITY:
            alerts = self._check_spatial_density_threshold(detections, threshold)
            
        elif threshold.threshold_type == ThresholdType.DATA_QUALITY:
            alerts = self._check_data_quality_threshold(detections, threshold)
            
        elif threshold.threshold_type == ThresholdType.TREND_CHANGE:
            alerts = self._check_trend_change_threshold(detections, threshold)
        
        return alerts
    
    def _check_emission_rate_threshold(self, detections: pd.DataFrame, 
                                     threshold: ThresholdConfig) -> List[Alert]:
        """Check emission rate thresholds."""
        
        if 'estimated_emission_rate_kg_hr' not in detections.columns:
            return []
        
        emission_rates = detections['estimated_emission_rate_kg_hr']
        violating_detections = self._apply_comparison(emission_rates, threshold)
        
        alerts = []
        for idx in violating_detections.index:
            detection = detections.loc[idx]
            
            alert = Alert(
                alert_id=f"EMIT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}",
                timestamp=datetime.now(),
                threshold_type=threshold.threshold_type,
                severity=threshold.alert_severity,
                emitter_id=detection.get('emitter_id'),
                value=detection['estimated_emission_rate_kg_hr'],
                threshold_value=threshold.threshold_value,
                message=f"Emission rate {detection['estimated_emission_rate_kg_hr']:.1f} kg/hr exceeds threshold {threshold.threshold_value} kg/hr",
                metadata={
                    'lat': detection.get('center_lat'),
                    'lon': detection.get('center_lon'),
                    'facility_name': detection.get('facility_name'),
                    'detection_time': detection.get('timestamp', datetime.now())
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_enhancement_threshold(self, detections: pd.DataFrame, 
                                   threshold: ThresholdConfig) -> List[Alert]:
        """Check methane enhancement thresholds."""
        
        if 'mean_enhancement' not in detections.columns:
            return []
        
        enhancements = detections['mean_enhancement']
        violating_detections = self._apply_comparison(enhancements, threshold)
        
        alerts = []
        for idx in violating_detections.index:
            detection = detections.loc[idx]
            
            alert = Alert(
                alert_id=f"ENHA_{datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}",
                timestamp=datetime.now(),
                threshold_type=threshold.threshold_type,
                severity=threshold.alert_severity,
                emitter_id=detection.get('emitter_id'),
                value=detection['mean_enhancement'],
                threshold_value=threshold.threshold_value,
                message=f"Enhancement {detection['mean_enhancement']:.1f} ppb exceeds threshold {threshold.threshold_value} ppb",
                metadata={
                    'lat': detection.get('center_lat'),
                    'lon': detection.get('center_lon'),
                    'emission_rate': detection.get('estimated_emission_rate_kg_hr')
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_detection_score_threshold(self, detections: pd.DataFrame, 
                                       threshold: ThresholdConfig) -> List[Alert]:
        """Check detection confidence score thresholds."""
        
        if 'detection_score' not in detections.columns:
            return []
        
        scores = detections['detection_score']
        violating_detections = self._apply_comparison(scores, threshold)
        
        alerts = []
        for idx in violating_detections.index:
            detection = detections.loc[idx]
            
            alert = Alert(
                alert_id=f"CONF_{datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}",
                timestamp=datetime.now(),
                threshold_type=threshold.threshold_type,
                severity=threshold.alert_severity,
                emitter_id=detection.get('emitter_id'),
                value=detection['detection_score'],
                threshold_value=threshold.threshold_value,
                message=f"High confidence detection (score: {detection['detection_score']:.3f})",
                metadata={
                    'lat': detection.get('center_lat'),
                    'lon': detection.get('center_lon'),
                    'emission_rate': detection.get('estimated_emission_rate_kg_hr')
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_spatial_density_threshold(self, detections: pd.DataFrame, 
                                       threshold: ThresholdConfig) -> List[Alert]:
        """Check spatial density of detections."""
        
        if len(detections) < threshold.min_detections:
            return []
        
        # Simple spatial clustering approach
        # In practice, you might use more sophisticated spatial analysis
        
        # Group detections by approximate location (0.1 degree grid)
        if 'center_lat' not in detections.columns or 'center_lon' not in detections.columns:
            return []
        
        lat_grid = np.round(detections['center_lat'] / 0.1) * 0.1
        lon_grid = np.round(detections['center_lon'] / 0.1) * 0.1
        
        location_groups = detections.groupby([lat_grid, lon_grid]).size()
        high_density_locations = location_groups[location_groups > threshold.threshold_value]
        
        alerts = []
        for (lat, lon), count in high_density_locations.items():
            alert = Alert(
                alert_id=f"DENS_{datetime.now().strftime('%Y%m%d%H%M%S')}_{lat}_{lon}",
                timestamp=datetime.now(),
                threshold_type=threshold.threshold_type,
                severity=threshold.alert_severity,
                emitter_id=None,
                value=count,
                threshold_value=threshold.threshold_value,
                message=f"High spatial density: {count} emitters in 0.1° grid cell",
                metadata={
                    'center_lat': lat,
                    'center_lon': lon,
                    'emitter_count': count
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_data_quality_threshold(self, detections: pd.DataFrame, 
                                    threshold: ThresholdConfig) -> List[Alert]:
        """Check data quality metrics."""
        
        # Calculate overall data quality score
        quality_metrics = []
        
        # Check for missing data
        if 'detection_score' in detections.columns:
            score_completeness = 1.0 - detections['detection_score'].isna().mean()
            quality_metrics.append(score_completeness)
        
        # Check for reasonable value ranges
        if 'estimated_emission_rate_kg_hr' in detections.columns:
            emission_rates = detections['estimated_emission_rate_kg_hr']
            reasonable_rates = ((emission_rates > 0) & (emission_rates < 50000)).mean()
            quality_metrics.append(reasonable_rates)
        
        # Overall quality score
        if quality_metrics:
            overall_quality = np.mean(quality_metrics)
            
            if self._compare_values(overall_quality, threshold.threshold_value, threshold.comparison_operator):
                alert = Alert(
                    alert_id=f"QUAL_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    timestamp=datetime.now(),
                    threshold_type=threshold.threshold_type,
                    severity=threshold.alert_severity,
                    emitter_id=None,
                    value=overall_quality,
                    threshold_value=threshold.threshold_value,
                    message=f"Low data quality detected: {overall_quality:.3f}",
                    metadata={
                        'quality_metrics': quality_metrics,
                        'detection_count': len(detections)
                    }
                )
                return [alert]
        
        return []
    
    def _check_trend_change_threshold(self, detections: pd.DataFrame, 
                                    threshold: ThresholdConfig) -> List[Alert]:
        """Check for significant trend changes (requires historical data)."""
        
        # This would require access to historical trend data
        # For now, return empty list as placeholder
        logger.debug("Trend change monitoring requires historical data integration")
        return []
    
    def _apply_comparison(self, values: pd.Series, threshold: ThresholdConfig) -> pd.Series:
        """Apply threshold comparison to values."""
        
        mask = self._compare_values(values, threshold.threshold_value, threshold.comparison_operator)
        return values[mask]
    
    def _compare_values(self, values: Union[float, pd.Series], threshold_value: float, 
                       operator: str) -> Union[bool, pd.Series]:
        """Compare values using specified operator."""
        
        if operator == '>':
            return values > threshold_value
        elif operator == '<':
            return values < threshold_value
        elif operator == '>=':
            return values >= threshold_value
        elif operator == '<=':
            return values <= threshold_value
        elif operator == '==':
            return values == threshold_value
        elif operator == '!=':
            return values != threshold_value
        else:
            raise ValueError(f"Unknown comparison operator: {operator}")
    
    def _apply_rate_limiting(self, alerts: List[Alert]) -> List[Alert]:
        """Apply rate limiting to prevent alert spam."""
        
        filtered_alerts = []
        current_time = datetime.now()
        
        for alert in alerts:
            # Create a key for rate limiting based on alert type and emitter
            rate_limit_key = f"{alert.threshold_type.value}_{alert.emitter_id}"
            
            # Check if enough time has passed since last similar alert
            if rate_limit_key in self.last_alert_times:
                time_since_last = current_time - self.last_alert_times[rate_limit_key]
                if time_since_last < self.rate_limit_window:
                    logger.debug(f"Rate limiting alert: {rate_limit_key}")
                    continue
            
            # Update last alert time and include alert
            self.last_alert_times[rate_limit_key] = current_time
            filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get currently active alerts."""
        
        if severity_filter:
            return [a for a in self.active_alerts if a.severity == severity_filter]
        return self.active_alerts.copy()
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.metadata['acknowledged_by'] = user
                alert.metadata['acknowledged_at'] = datetime.now()
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
        
        logger.warning(f"Alert {alert_id} not found for acknowledgment")
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "system", resolution_note: str = "") -> bool:
        """Resolve an alert."""
        
        for i, alert in enumerate(self.active_alerts):
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.metadata['resolved_by'] = user
                alert.metadata['resolved_at'] = datetime.now()
                alert.metadata['resolution_note'] = resolution_note
                
                # Remove from active alerts
                self.active_alerts.pop(i)
                
                logger.info(f"Alert {alert_id} resolved by {user}")
                return True
        
        logger.warning(f"Alert {alert_id} not found for resolution")
        return False
    
    def get_alert_statistics(self) -> Dict:
        """Get alert statistics."""
        
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        # Count by severity
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                a for a in self.alert_history if a.severity == severity
            ])
        
        # Count by type
        type_counts = {}
        for threshold_type in ThresholdType:
            type_counts[threshold_type.value] = len([
                a for a in self.alert_history if a.threshold_type == threshold_type
            ])
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'resolved_alerts': total_alerts - active_alerts,
            'severity_distribution': severity_counts,
            'type_distribution': type_counts,
            'configured_thresholds': len(self.thresholds)
        }
    
    def export_alerts_to_dataframe(self) -> pd.DataFrame:
        """Export alert history to DataFrame."""
        
        if not self.alert_history:
            return pd.DataFrame()
        
        alert_records = []
        for alert in self.alert_history:
            record = {
                'alert_id': alert.alert_id,
                'timestamp': alert.timestamp,
                'threshold_type': alert.threshold_type.value,
                'severity': alert.severity.value,
                'emitter_id': alert.emitter_id,
                'value': alert.value,
                'threshold_value': alert.threshold_value,
                'message': alert.message,
                'acknowledged': alert.acknowledged,
                'resolved': alert.resolved
            }
            
            # Add metadata fields
            for key, value in alert.metadata.items():
                record[f'meta_{key}'] = value
            
            alert_records.append(record)
        
        return pd.DataFrame(alert_records)