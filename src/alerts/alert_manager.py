# ============================================================================
# FILE: src/alerts/alert_manager.py
# ============================================================================
import logging
import smtplib
import requests
import json
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Comprehensive alert management system for super-emitter tracking.
    
    Features:
    - Multi-channel notification (email, webhook, dashboard)
    - Alert prioritization and filtering
    - Alert history and tracking
    - Customizable thresholds and conditions
    - Integration with external systems
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_config = config['alerts']
        self.alert_history = []
        self.active_alerts = []
        
        logger.info("AlertManager initialized")
    
    def process_alerts(self, raw_alerts: List[Dict]) -> Dict:
        """
        Process and prioritize alerts from various sources.
        
        Args:
            raw_alerts: List of alert dictionaries from detection/tracking
            
        Returns:
            Dictionary with processed alerts and summary
        """
        
        logger.info(f"Processing {len(raw_alerts)} raw alerts")
        
        # Filter and validate alerts
        valid_alerts = self._validate_alerts(raw_alerts)
        
        # Deduplicate alerts
        deduplicated_alerts = self._deduplicate_alerts(valid_alerts)
        
        # Prioritize and score alerts
        scored_alerts = self._score_alerts(deduplicated_alerts)
        
        # Apply alert conditions and thresholds
        filtered_alerts = self._apply_alert_conditions(scored_alerts)
        
        # Update alert history
        self._update_alert_history(filtered_alerts)
        
        # Generate alert summary
        alert_summary = self._generate_alert_summary(filtered_alerts)
        
        processed_alerts = {
            'alerts': filtered_alerts,
            'summary': alert_summary,
            'processing_timestamp': datetime.now(),
            'total_processed': len(raw_alerts),
            'alerts_generated': len(filtered_alerts)
        }
        
        logger.info(f"Generated {len(filtered_alerts)} actionable alerts")
        return processed_alerts
    
    def _validate_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Validate alert format and required fields."""
        
        required_fields = ['alert_type', 'severity', 'message']
        valid_alerts = []
        
        for alert in alerts:
            # Check required fields
            if all(field in alert for field in required_fields):
                # Standardize severity levels
                alert['severity'] = self._standardize_severity(alert['severity'])
                
                # Add timestamp if missing
                if 'timestamp' not in alert:
                    alert['timestamp'] = datetime.now()
                
                # Add unique alert ID
                alert['alert_id'] = f"ALT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(valid_alerts):04d}"
                
                valid_alerts.append(alert)
            else:
                logger.warning(f"Invalid alert format: {alert}")
        
        return valid_alerts
    
    def _standardize_severity(self, severity: str) -> str:
        """Standardize severity levels."""
        
        severity_map = {
            'critical': 'high',
            'major': 'high',
            'warning': 'medium',
            'moderate': 'medium',
            'minor': 'low',
            'info': 'low',
            'informational': 'low'
        }
        
        severity_lower = severity.lower()
        return severity_map.get(severity_lower, severity_lower)
    
    def _deduplicate_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Remove duplicate alerts based on similarity."""
        
        if not alerts:
            return alerts
        
        deduplicated = []
        
        for alert in alerts:
            # Check for duplicates based on key fields
            is_duplicate = False
            
            for existing in deduplicated:
                if self._alerts_are_similar(alert, existing):
                    # Update existing alert with latest information
                    existing['count'] = existing.get('count', 1) + 1
                    existing['last_occurrence'] = alert['timestamp']
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                alert['count'] = 1
                alert['first_occurrence'] = alert['timestamp']
                alert['last_occurrence'] = alert['timestamp']
                deduplicated.append(alert)
        
        logger.info(f"Deduplicated {len(alerts)} alerts to {len(deduplicated)}")
        return deduplicated
    
    def _alerts_are_similar(self, alert1: Dict, alert2: Dict) -> bool:
        """Check if two alerts are similar enough to be considered duplicates."""
        
        # Same alert type and same emitter
        if (alert1.get('alert_type') == alert2.get('alert_type') and
            alert1.get('emitter_id') == alert2.get('emitter_id')):
            
            # Within time window
            time_diff = abs((alert1['timestamp'] - alert2['last_occurrence']).total_seconds())
            if time_diff < 3600:  # 1 hour window
                return True
        
        return False
    
    def _score_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Score alerts based on importance and urgency."""
        
        for alert in alerts:
            score = 0
            
            # Base score from severity
            severity_scores = {'high': 80, 'medium': 50, 'low': 20}
            score += severity_scores.get(alert['severity'], 20)
            
            # Alert type importance
            type_scores = {
                'new_super_emitter': 15,
                'emission_increase': 12,
                'missing_emitter': 10,
                'data_quality': 5
            }
            score += type_scores.get(alert['alert_type'], 5)
            
            # Facility association bonus
            if alert.get('facility_info', {}).get('facility_id'):
                score += 10
            
            # Emission magnitude
            emission_rate = alert.get('current_emission_rate', 0)
            if emission_rate > 2000:
                score += 15
            elif emission_rate > 1000:
                score += 10
            elif emission_rate > 500:
                score += 5
            
            # Trend significance
            p_value = alert.get('p_value', 1.0)
            if p_value < 0.001:
                score += 10
            elif p_value < 0.01:
                score += 5
            
            # Frequency penalty for repeated alerts
            if alert.get('count', 1) > 1:
                score -= min(20, alert['count'] * 2)
            
            alert['alert_score'] = max(0, min(100, score))
        
        # Sort by score
        alerts.sort(key=lambda x: x['alert_score'], reverse=True)
        
        return alerts
    
    def _apply_alert_conditions(self, alerts: List[Dict]) -> List[Dict]:
        """Apply configured alert conditions and thresholds."""
        
        conditions = self.alert_config['conditions']
        filtered_alerts = []
        
        for alert in alerts:
            should_alert = False
            
            # Check specific conditions
            alert_type = alert['alert_type']
            
            if alert_type == 'new_super_emitter' and conditions['new_super_emitter']:
                confidence = alert.get('detection_score', 0)
                threshold = self.alert_config['thresholds']['new_emitter_confidence']
                should_alert = confidence >= threshold
            
            elif alert_type == 'emission_increase' and conditions['emission_increase']:
                change_percent = alert.get('change_percent', 0)
                threshold = self.alert_config['thresholds']['emission_increase_percent']
                should_alert = change_percent >= threshold
            
            elif alert_type == 'missing_emitter' and conditions['facility_shutdown']:
                days_missing = alert.get('days_missing', 0)
                should_alert = days_missing > 7  # Alert after 7 days
            
            elif alert_type == 'data_quality' and conditions['data_quality_issues']:
                should_alert = True  # Always alert on data quality issues
            
            # Additional filters
            if should_alert:
                # Minimum alert score threshold
                if alert['alert_score'] >= 30:
                    filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def _update_alert_history(self, alerts: List[Dict]):
        """Update alert history for tracking and analysis."""
        
        for alert in alerts:
            # Add to history
            history_entry = {
                'alert_id': alert['alert_id'],
                'timestamp': alert['timestamp'],
                'alert_type': alert['alert_type'],
                'severity': alert['severity'],
                'alert_score': alert['alert_score'],
                'emitter_id': alert.get('emitter_id'),
                'processed_timestamp': datetime.now(),
                'status': 'active'
            }
            
            self.alert_history.append(history_entry)
        
        # Keep only recent history (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        self.alert_history = [
            h for h in self.alert_history 
            if h['timestamp'] >= cutoff_date
        ]
        
        # Update active alerts
        self.active_alerts = [a for a in alerts if a['severity'] in ['high', 'medium']]
    
    def _generate_alert_summary(self, alerts: List[Dict]) -> Dict:
        """Generate summary statistics for alerts."""
        
        if not alerts:
            return {
                'total_alerts': 0,
                'by_severity': {'high': 0, 'medium': 0, 'low': 0},
                'by_type': {},
                'high_priority_count': 0,
                'requires_immediate_attention': 0
            }
        
        summary = {
            'total_alerts': len(alerts),
            'high_priority_count': len([a for a in alerts if a['severity'] == 'high']),
            'requires_immediate_attention': len([a for a in alerts if a['alert_score'] > 80])
        }
        
        # Count by severity
        summary['by_severity'] = {
            'high': len([a for a in alerts if a['severity'] == 'high']),
            'medium': len([a for a in alerts if a['severity'] == 'medium']),
            'low': len([a for a in alerts if a['severity'] == 'low'])
        }
        
        # Count by type
        summary['by_type'] = {}
        for alert in alerts:
            alert_type = alert['alert_type']
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
        
        return summary
    
    def send_email_notifications(self, processed_alerts: Dict) -> bool:
        """Send email notifications for high-priority alerts."""
        
        email_config = self.alert_config['notifications']['email']
        
        if not email_config['enabled']:
            logger.info("Email notifications disabled")
            return True
        
        high_priority_alerts = [
            alert for alert in processed_alerts['alerts'] 
            if alert['severity'] == 'high'
        ]
        
        if not high_priority_alerts:
            logger.info("No high-priority alerts to send")
            return True
        
        try:
            # Compose email
            subject = f"Super-Emitter Alert: {len(high_priority_alerts)} High Priority Alert(s)"
            body = self._compose_alert_email(high_priority_alerts, processed_alerts['summary'])
            
            # Send email
            success = self._send_email(
                recipients=email_config['recipients'],
                subject=subject,
                body=body
            )
            
            if success:
                logger.info(f"Email notification sent for {len(high_priority_alerts)} alerts")
            else:
                logger.error("Failed to send email notification")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    def _compose_alert_email(self, alerts: List[Dict], summary: Dict) -> str:
        """Compose email body for alert notifications."""
        
        body = f"""
Super-Emitter Alert Notification
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

SUMMARY:
- Total High Priority Alerts: {len(alerts)}
- Immediate Attention Required: {summary['requires_immediate_attention']}

HIGH PRIORITY ALERTS:
{'='*50}
"""
        
        for i, alert in enumerate(alerts, 1):
            emitter_info = f"Emitter: {alert.get('emitter_id', 'Unknown')}"
            facility_info = ""
            
            if alert.get('facility_info', {}).get('facility_name'):
                facility_info = f"Facility: {alert['facility_info']['facility_name']}"
            
            body += f"""
{i}. {alert['alert_type'].upper()}
   {emitter_info}
   {facility_info}
   Message: {alert['message']}
   Score: {alert['alert_score']}/100
   Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
   
"""
        
        body += f"""
{'='*50}

For detailed analysis, please access the monitoring dashboard.

This is an automated alert from the Super-Emitter Tracking System.
"""
        
        return body
    
    def _send_email(self, recipients: List[str], subject: str, body: str) -> bool:
        """Send email using SMTP configuration."""
        
        email_config = self.alert_config['notifications']['email']
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = email_config.get('sender', 'noreply@superemitter.system')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Attach body
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(
                email_config['smtp_server'], 
                email_config['smtp_port']
            )
            
            if email_config.get('use_tls', True):
                server.starttls()
            
            if email_config.get('username') and email_config.get('password'):
                server.login(email_config['username'], email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"SMTP error: {e}")
            return False
    
    def send_webhook_notifications(self, processed_alerts: Dict) -> bool:
        """Send webhook notifications for integration with external systems."""
        
        webhook_config = self.alert_config['notifications']['webhook']
        
        if not webhook_config['enabled']:
            logger.info("Webhook notifications disabled")
            return True
        
        try:
            # Prepare webhook payload
            payload = {
                'timestamp': datetime.now().isoformat(),
                'source': 'super-emitter-tracking-system',
                'alert_summary': processed_alerts['summary'],
                'alerts': processed_alerts['alerts']
            }
            
            # Send webhook
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("Webhook notification sent successfully")
                return True
            else:
                logger.error(f"Webhook failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return False
    
    def generate_new_emitter_alerts(self, detections: pd.DataFrame) -> List[Dict]:
        """Generate alerts for newly detected super-emitters."""
        
        alerts = []
        
        if len(detections) == 0:
            return alerts
        
        # Filter for new emitters (those without facility associations might be new)
        potential_new = detections[detections['facility_id'].isna()]
        
        for _, emitter in potential_new.iterrows():
            emission_rate = emitter.get('estimated_emission_rate_kg_hr', 0)
            detection_score = emitter.get('detection_score', 0)
            
            # Check if this qualifies as a significant new emitter
            if (emission_rate > self.alert_config['thresholds']['emission_increase_percent'] and
                detection_score > self.alert_config['thresholds']['new_emitter_confidence']):
                
                alert = {
                    'alert_type': 'new_super_emitter',
                    'severity': 'high' if emission_rate > 2000 else 'medium',
                    'emitter_id': emitter.get('emitter_id'),
                    'timestamp': datetime.now(),
                    'message': f"New super-emitter detected with {emission_rate:.0f} kg/hr emission rate",
                    'current_emission_rate': emission_rate,
                    'detection_score': detection_score,
                    'location': {
                        'lat': emitter.get('center_lat'),
                        'lon': emitter.get('center_lon')
                    },
                    'facility_info': {
                        'facility_id': emitter.get('facility_id'),
                        'facility_name': emitter.get('facility_name'),
                        'facility_type': emitter.get('facility_type')
                    }
                }
                alerts.append(alert)
        
        return alerts
    
    def get_alert_history(self, days: int = 7) -> pd.DataFrame:
        """Get alert history for the specified number of days."""
        
        if not self.alert_history:
            return pd.DataFrame()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert['timestamp'] >= cutoff_date
        ]
        
        return pd.DataFrame(recent_alerts)
    
    def get_active_alerts(self) -> List[Dict]:
        """Get currently active alerts."""
        return self.active_alerts.copy()
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        
        for alert in self.active_alerts:
            if alert['alert_id'] == alert_id:
                alert['status'] = 'acknowledged'
                alert['acknowledged_by'] = user
                alert['acknowledged_at'] = datetime.now()
                
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
        
        logger.warning(f"Alert {alert_id} not found in active alerts")
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "system", 
                     resolution_note: str = "") -> bool:
        """Resolve an alert."""
        
        for i, alert in enumerate(self.active_alerts):
            if alert['alert_id'] == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_by'] = user
                alert['resolved_at'] = datetime.now()
                alert['resolution_note'] = resolution_note
                
                # Move to history and remove from active
                self.alert_history.append(alert)
                self.active_alerts.pop(i)
                
                logger.info(f"Alert {alert_id} resolved by {user}")
                return True
        
        logger.warning(f"Alert {alert_id} not found in active alerts")
        return False
    
    def get_alert_statistics(self) -> Dict:
        """Get comprehensive alert statistics."""
        
        if not self.alert_history:
            return {'total_alerts': 0}
        
        df = pd.DataFrame(self.alert_history)
        
        stats = {
            'total_alerts': len(df),
            'active_alerts': len(self.active_alerts),
            'alerts_last_24h': len(df[df['timestamp'] > datetime.now() - timedelta(days=1)]),
            'alerts_last_7d': len(df[df['timestamp'] > datetime.now() - timedelta(days=7)]),
            'by_severity': df['severity'].value_counts().to_dict(),
            'by_type': df['alert_type'].value_counts().to_dict(),
            'average_score': float(df['alert_score'].mean()) if 'alert_score' in df.columns else 0,
            'resolution_rate': len(df[df['status'] == 'resolved']) / len(df) if len(df) > 0 else 0
        }
        
        return stats