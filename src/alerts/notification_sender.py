# ============================================================================
# FILE: src/alerts/notification_sender.py
# ============================================================================
import smtplib
import requests
import json
import logging
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pathlib import Path
import os
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
import warnings

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Types of notifications."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"

class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    notification_type: NotificationType
    enabled: bool
    config: Dict[str, Any]
    priority_filter: Optional[NotificationPriority] = None
    rate_limit_minutes: int = 0

@dataclass
class Notification:
    """Notification data structure."""
    notification_id: str
    timestamp: datetime
    notification_type: NotificationType
    priority: NotificationPriority
    subject: str
    message: str
    recipients: List[str]
    attachments: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    sent: bool = False
    error_message: Optional[str] = None

class NotificationSender:
    """
    Send notifications through multiple channels for super-emitter alerts.
    
    Features:
    - Multiple notification channels (email, webhook, SMS, Slack, etc.)
    - Priority-based filtering and routing
    - Rate limiting and batching
    - Template-based message formatting
    - Delivery confirmation and error handling
    - Asynchronous sending for performance
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.notification_config = config.get('alerts', {}).get('notifications', {})
        
        # Notification channels
        self.channels: List[NotificationConfig] = []
        self.notification_queue: List[Notification] = []
        self.notification_history: List[Notification] = []
        
        # Rate limiting
        self.last_sent_times: Dict[str, datetime] = {}
        
        # Message templates
        self.templates = self._load_message_templates()
        
        self._initialize_notification_channels()
        
        logger.info("NotificationSender initialized")
    
    def _initialize_notification_channels(self):
        """Initialize notification channels from configuration."""
        
        # Email configuration
        email_config = self.notification_config.get('email', {})
        if email_config.get('enabled', False):
            self.add_notification_channel(NotificationConfig(
                notification_type=NotificationType.EMAIL,
                enabled=True,
                config=email_config,
                priority_filter=None,
                rate_limit_minutes=email_config.get('rate_limit_minutes', 5)
            ))
        
        # Webhook configuration
        webhook_config = self.notification_config.get('webhook', {})
        if webhook_config.get('enabled', False):
            self.add_notification_channel(NotificationConfig(
                notification_type=NotificationType.WEBHOOK,
                enabled=True,
                config=webhook_config,
                priority_filter=None,
                rate_limit_minutes=webhook_config.get('rate_limit_minutes', 1)
            ))
        
        # Slack configuration
        slack_config = self.notification_config.get('slack', {})
        if slack_config.get('enabled', False):
            self.add_notification_channel(NotificationConfig(
                notification_type=NotificationType.SLACK,
                enabled=True,
                config=slack_config,
                priority_filter=NotificationPriority.HIGH,  # Only high priority to Slack
                rate_limit_minutes=slack_config.get('rate_limit_minutes', 10)
            ))
    
    def add_notification_channel(self, channel_config: NotificationConfig):
        """Add a notification channel."""
        self.channels.append(channel_config)
        logger.info(f"Added notification channel: {channel_config.notification_type.value}")
    
    def send_alert_notification(self, alert_data: Dict, 
                              priority: NotificationPriority = NotificationPriority.NORMAL,
                              custom_message: Optional[str] = None) -> List[str]:
        """
        Send notifications for an alert.
        
        Args:
            alert_data: Alert information dictionary
            priority: Notification priority
            custom_message: Custom message override
            
        Returns:
            List of notification IDs that were sent
        """
        
        # Create notification message
        if custom_message:
            message = custom_message
        else:
            message = self._format_alert_message(alert_data)
        
        subject = self._format_alert_subject(alert_data)
        
        # Create notification
        notification = Notification(
            notification_id=f"NOTIF_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.notification_history)}",
            timestamp=datetime.now(),
            notification_type=NotificationType.EMAIL,  # Will be overridden for each channel
            priority=priority,
            subject=subject,
            message=message,
            recipients=self._get_recipients_for_alert(alert_data, priority),
            metadata={'alert_data': alert_data}
        )
        
        # Send through all appropriate channels
        sent_notifications = []
        for channel in self.channels:
            if not channel.enabled:
                continue
            
            # Check priority filter
            if channel.priority_filter and priority.value != channel.priority_filter.value:
                continue
            
            # Check rate limiting
            if self._is_rate_limited(channel):
                logger.debug(f"Rate limited: {channel.notification_type.value}")
                continue
            
            # Create channel-specific notification
            channel_notification = self._create_channel_notification(notification, channel)
            
            # Send notification
            try:
                success = self._send_notification(channel_notification, channel)
                if success:
                    sent_notifications.append(channel_notification.notification_id)
                    self._update_rate_limit(channel)
                    self.notification_history.append(channel_notification)
                
            except Exception as e:
                logger.error(f"Failed to send {channel.notification_type.value} notification: {e}")
                channel_notification.error_message = str(e)
                self.notification_history.append(channel_notification)
        
        return sent_notifications
    
    def send_summary_notification(self, summary_data: Dict,
                                recipients: Optional[List[str]] = None) -> List[str]:
        """Send summary notification (e.g., daily/weekly reports)."""
        
        message = self._format_summary_message(summary_data)
        subject = f"Super-Emitter Tracking Summary - {datetime.now().strftime('%Y-%m-%d')}"
        
        if not recipients:
            recipients = self._get_default_recipients()
        
        notification = Notification(
            notification_id=f"SUMM_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            notification_type=NotificationType.EMAIL,
            priority=NotificationPriority.NORMAL,
            subject=subject,
            message=message,
            recipients=recipients,
            metadata={'summary_data': summary_data}
        )
        
        # Send through email channel (summaries typically go via email)
        email_channel = next((c for c in self.channels if c.notification_type == NotificationType.EMAIL), None)
        
        if email_channel and email_channel.enabled:
            try:
                success = self._send_notification(notification, email_channel)
                if success:
                    self.notification_history.append(notification)
                    return [notification.notification_id]
            except Exception as e:
                logger.error(f"Failed to send summary notification: {e}")
        
        return []
    
    def _send_notification(self, notification: Notification, 
                          channel: NotificationConfig) -> bool:
        """Send notification through specific channel."""
        
        if channel.notification_type == NotificationType.EMAIL:
            return self._send_email(notification, channel.config)
        
        elif channel.notification_type == NotificationType.WEBHOOK:
            return self._send_webhook(notification, channel.config)
        
        elif channel.notification_type == NotificationType.SLACK:
            return self._send_slack(notification, channel.config)
        
        elif channel.notification_type == NotificationType.SMS:
            return self._send_sms(notification, channel.config)
        
        else:
            logger.warning(f"Unsupported notification type: {channel.notification_type.value}")
            return False
    
    def _send_email(self, notification: Notification, email_config: Dict) -> bool:
        """Send email notification."""
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = email_config.get('sender', 'noreply@superemitter.system')
            msg['To'] = ', '.join(notification.recipients)
            msg['Subject'] = notification.subject
            
            # Add body
            body = MimeText(notification.message, 'html' if '<html>' in notification.message else 'plain')
            msg.attach(body)
            
            # Add attachments if any
            if notification.attachments:
                for file_path in notification.attachments:
                    if Path(file_path).exists():
                        with open(file_path, 'rb') as attachment:
                            part = MimeBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {Path(file_path).name}'
                            )
                            msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            
            if email_config.get('use_tls', True):
                server.starttls()
            
            if email_config.get('username') and email_config.get('password'):
                server.login(email_config['username'], email_config['password'])
            
            text = msg.as_string()
            server.sendmail(msg['From'], notification.recipients, text)
            server.quit()
            
            notification.sent = True
            logger.info(f"Email sent to {len(notification.recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            notification.error_message = str(e)
            return False
    
    def _send_webhook(self, notification: Notification, webhook_config: Dict) -> bool:
        """Send webhook notification."""
        
        try:
            url = webhook_config['url']
            
            # Prepare payload
            payload = {
                'notification_id': notification.notification_id,
                'timestamp': notification.timestamp.isoformat(),
                'priority': notification.priority.value,
                'subject': notification.subject,
                'message': notification.message,
                'metadata': notification.metadata or {}
            }
            
            # Add custom headers if configured
            headers = {'Content-Type': 'application/json'}
            if 'headers' in webhook_config:
                headers.update(webhook_config['headers'])
            
            # Send request
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=webhook_config.get('timeout', 30)
            )
            
            if response.status_code == 200:
                notification.sent = True
                logger.info(f"Webhook sent successfully to {url}")
                return True
            else:
                logger.error(f"Webhook failed with status {response.status_code}: {response.text}")
                notification.error_message = f"HTTP {response.status_code}: {response.text}"
                return False
                
        except Exception as e:
            logger.error(f"Webhook sending failed: {e}")
            notification.error_message = str(e)
            return False
    
    def _send_slack(self, notification: Notification, slack_config: Dict) -> bool:
        """Send Slack notification."""
        
        try:
            webhook_url = slack_config['webhook_url']
            
            # Format message for Slack
            slack_message = {
                "text": notification.subject,
                "attachments": [
                    {
                        "color": self._get_slack_color(notification.priority),
                        "fields": [
                            {
                                "title": "Priority",
                                "value": notification.priority.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": notification.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            }
                        ],
                        "text": notification.message
                    }
                ]
            }
            
            # Add alert-specific fields if available
            if notification.metadata and 'alert_data' in notification.metadata:
                alert_data = notification.metadata['alert_data']
                
                if 'emitter_id' in alert_data:
                    slack_message["attachments"][0]["fields"].append({
                        "title": "Emitter ID",
                        "value": alert_data['emitter_id'],
                        "short": True
                    })
                
                if 'estimated_emission_rate_kg_hr' in alert_data:
                    slack_message["attachments"][0]["fields"].append({
                        "title": "Emission Rate",
                        "value": f"{alert_data['estimated_emission_rate_kg_hr']:.1f} kg/hr",
                        "short": True
                    })
            
            # Send to Slack
            response = requests.post(webhook_url, json=slack_message, timeout=30)
            
            if response.status_code == 200:
                notification.sent = True
                logger.info("Slack notification sent successfully")
                return True
            else:
                logger.error(f"Slack notification failed: {response.text}")
                notification.error_message = response.text
                return False
                
        except Exception as e:
            logger.error(f"Slack sending failed: {e}")
            notification.error_message = str(e)
            return False
    
    def _send_sms(self, notification: Notification, sms_config: Dict) -> bool:
        """Send SMS notification (placeholder - requires SMS service integration)."""
        
        logger.warning("SMS notifications not implemented - requires SMS service integration")
        # Would integrate with services like Twilio, AWS SNS, etc.
        
        return False
    
    def _get_slack_color(self, priority: NotificationPriority) -> str:
        """Get Slack attachment color based on priority."""
        
        color_map = {
            NotificationPriority.LOW: "#36a64f",      # Green
            NotificationPriority.NORMAL: "#2196F3",   # Blue
            NotificationPriority.HIGH: "#ff9800",     # Orange
            NotificationPriority.URGENT: "#f44336"    # Red
        }
        
        return color_map.get(priority, "#2196F3")
    
    def _format_alert_message(self, alert_data: Dict) -> str:
        """Format alert message using templates."""
        
        template_name = alert_data.get('alert_type', 'default')
        template = self.templates.get(template_name, self.templates['default'])
        
        try:
            # Format template with alert data
            message = template.format(**alert_data)
            return message
        except KeyError as e:
            logger.warning(f"Template formatting failed for key {e}, using default")
            return self.templates['default'].format(
                alert_type=alert_data.get('alert_type', 'Unknown'),
                message=alert_data.get('message', 'Alert triggered'),
                timestamp=alert_data.get('timestamp', datetime.now())
            )
    
    def _format_alert_subject(self, alert_data: Dict) -> str:
        """Format alert subject line."""
        
        severity = alert_data.get('severity', 'medium').upper()
        alert_type = alert_data.get('alert_type', 'Alert')
        
        return f"[{severity}] Super-Emitter {alert_type.replace('_', ' ').title()}"
    
    def _format_summary_message(self, summary_data: Dict) -> str:
        """Format summary message."""
        
        template = """
        <html>
        <body>
        <h2>Super-Emitter Tracking Summary</h2>
        <p>Generated: {timestamp}</p>
        
        <h3>Key Metrics</h3>
        <ul>
            <li>Total Active Emitters: {total_emitters}</li>
            <li>Total Emission Rate: {total_emission_rate:.1f} kg/hr</li>
            <li>New Detections: {new_detections}</li>
            <li>Alerts Generated: {alerts_generated}</li>
        </ul>
        
        <h3>Top Emitters</h3>
        {top_emitters_table}
        
        <h3>Alert Summary</h3>
        <ul>
            <li>High Priority: {high_priority_alerts}</li>
            <li>Medium Priority: {medium_priority_alerts}</li>
            <li>Low Priority: {low_priority_alerts}</li>
        </ul>
        
        <p>For detailed analysis, please access the monitoring dashboard.</p>
        </body>
        </html>
        """
        
        # Create top emitters table
        top_emitters = summary_data.get('top_emitters', [])
        if top_emitters:
            table_rows = []
            for emitter in top_emitters[:5]:  # Top 5
                table_rows.append(
                    f"<tr><td>{emitter.get('emitter_id', 'Unknown')}</td>"
                    f"<td>{emitter.get('emission_rate', 0):.1f} kg/hr</td></tr>"
                )
            top_emitters_table = f"<table border='1'><tr><th>Emitter</th><th>Rate</th></tr>{''.join(table_rows)}</table>"
        else:
            top_emitters_table = "<p>No emitter data available</p>"
        
        return template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_emitters=summary_data.get('total_emitters', 0),
            total_emission_rate=summary_data.get('total_emission_rate', 0),
            new_detections=summary_data.get('new_detections', 0),
            alerts_generated=summary_data.get('alerts_generated', 0),
            top_emitters_table=top_emitters_table,
            high_priority_alerts=summary_data.get('high_priority_alerts', 0),
            medium_priority_alerts=summary_data.get('medium_priority_alerts', 0),
            low_priority_alerts=summary_data.get('low_priority_alerts', 0)
        )
    
    def _load_message_templates(self) -> Dict[str, str]:
        """Load message templates for different alert types."""
        
        return {
            'default': """
Alert Type: {alert_type}
Message: {message}
Time: {timestamp}

This is an automated alert from the Super-Emitter Tracking System.
            """,
            
            'new_super_emitter': """
🚨 NEW SUPER-EMITTER DETECTED 🚨

Emitter ID: {emitter_id}
Location: {center_lat:.3f}°N, {center_lon:.3f}°W
Emission Rate: {estimated_emission_rate_kg_hr:.1f} kg/hr
Enhancement: {mean_enhancement:.1f} ppb
Detection Score: {detection_score:.3f}

Facility: {facility_name}
Time: {timestamp}

This emitter requires immediate attention for verification and potential mitigation actions.
            """,
            
            'emission_increase': """
📈 EMISSION RATE INCREASE DETECTED

Emitter ID: {emitter_id}
Current Rate: {current_emission_rate:.1f} kg/hr
Previous Rate: {previous_emission_rate:.1f} kg/hr
Change: +{change_percent:.1f}%

Location: {center_lat:.3f}°N, {center_lon:.3f}°W
Time: {timestamp}

Significant increase in emission rate detected. Investigation recommended.
            """,
            
            'missing_emitter': """
⚠️ EMITTER NO LONGER DETECTED

Emitter ID: {emitter_id}
Last Detection: {last_detection}
Days Missing: {days_missing}
Previous Rate: {mean_emission_rate:.1f} kg/hr

Location: {center_lat:.3f}°N, {center_lon:.3f}°W

This emitter has not been detected for {days_missing} days. Possible shutdown or data quality issue.
            """,
            
            'data_quality': """
⚠️ DATA QUALITY ISSUE

Issue Type: {quality_issue_type}
Affected Region: {affected_region}
Data Completeness: {data_completeness:.1%}
Time: {timestamp}

Data quality has degraded in the monitoring region. This may affect detection accuracy.
            """
        }
    
    def _get_recipients_for_alert(self, alert_data: Dict, priority: NotificationPriority) -> List[str]:
        """Get appropriate recipients based on alert type and priority."""
        
        # Get default recipients from config
        default_recipients = self.notification_config.get('email', {}).get('recipients', [])
        
        # Add priority-specific recipients
        if priority == NotificationPriority.URGENT:
            urgent_recipients = self.notification_config.get('urgent_recipients', [])
            return list(set(default_recipients + urgent_recipients))
        
        elif priority == NotificationPriority.HIGH:
            high_priority_recipients = self.notification_config.get('high_priority_recipients', [])
            return list(set(default_recipients + high_priority_recipients))
        
        return default_recipients
    
    def _get_default_recipients(self) -> List[str]:
        """Get default recipients for summary notifications."""
        return self.notification_config.get('email', {}).get('recipients', [])
    
    def _create_channel_notification(self, base_notification: Notification, 
                                   channel: NotificationConfig) -> Notification:
        """Create channel-specific notification."""
        
        # Create copy with channel-specific settings
        channel_notification = Notification(
            notification_id=f"{base_notification.notification_id}_{channel.notification_type.value}",
            timestamp=base_notification.timestamp,
            notification_type=channel.notification_type,
            priority=base_notification.priority,
            subject=base_notification.subject,
            message=base_notification.message,
            recipients=base_notification.recipients,
            attachments=base_notification.attachments,
            metadata=base_notification.metadata
        )
        
        return channel_notification
    
    def _is_rate_limited(self, channel: NotificationConfig) -> bool:
        """Check if channel is rate limited."""
        
        if channel.rate_limit_minutes <= 0:
            return False
        
        channel_key = channel.notification_type.value
        
        if channel_key in self.last_sent_times:
            time_since_last = datetime.now() - self.last_sent_times[channel_key]
            return time_since_last.total_seconds() < (channel.rate_limit_minutes * 60)
        
        return False
    
    def _update_rate_limit(self, channel: NotificationConfig):
        """Update rate limit timestamp for channel."""
        
        channel_key = channel.notification_type.value
        self.last_sent_times[channel_key] = datetime.now()
    
    def get_notification_statistics(self) -> Dict:
        """Get notification statistics."""
        
        total_notifications = len(self.notification_history)
        successful_notifications = len([n for n in self.notification_history if n.sent])
        
        # Count by type
        type_counts = {}
        for notification_type in NotificationType:
            type_counts[notification_type.value] = len([
                n for n in self.notification_history if n.notification_type == notification_type
            ])
        
        # Count by priority
        priority_counts = {}
        for priority in NotificationPriority:
            priority_counts[priority.value] = len([
                n for n in self.notification_history if n.priority == priority
            ])
        
        return {
            'total_notifications': total_notifications,
            'successful_notifications': successful_notifications,
            'failed_notifications': total_notifications - successful_notifications,
            'success_rate': successful_notifications / total_notifications if total_notifications > 0 else 0,
            'type_distribution': type_counts,
            'priority_distribution': priority_counts,
            'configured_channels': len(self.channels)
        }
    
    def test_notification_channels(self) -> Dict[str, bool]:
        """Test all configured notification channels."""
        
        test_results = {}
        
        test_notification = Notification(
            notification_id=f"TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            notification_type=NotificationType.EMAIL,  # Will be overridden
            priority=NotificationPriority.LOW,
            subject="Super-Emitter System Test Notification",
            message="This is a test notification to verify the notification system is working correctly.",
            recipients=self._get_default_recipients()
        )
        
        for channel in self.channels:
            if not channel.enabled:
                test_results[channel.notification_type.value] = False
                continue
            
            try:
                channel_notification = self._create_channel_notification(test_notification, channel)
                success = self._send_notification(channel_notification, channel)
                test_results[channel.notification_type.value] = success
                
            except Exception as e:
                logger.error(f"Test failed for {channel.notification_type.value}: {e}")
                test_results[channel.notification_type.value] = False
        
        return test_results
    
    async def send_bulk_notifications(self, notifications: List[Notification]) -> List[str]:
        """Send multiple notifications asynchronously."""
        
        # This would implement async sending for better performance
        # For now, using synchronous approach
        
        sent_ids = []
        for notification in notifications:
            # Convert notification data to alert format for existing method
            alert_data = notification.metadata.get('alert_data', {})
            alert_data.update({
                'message': notification.message,
                'timestamp': notification.timestamp
            })
            
            try:
                ids = self.send_alert_notification(alert_data, notification.priority)
                sent_ids.extend(ids)
            except Exception as e:
                logger.error(f"Bulk notification failed: {e}")
        
        return sent_ids