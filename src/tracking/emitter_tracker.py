# ============================================================================
# FILE: src/tracking/emitter_tracker.py
# ============================================================================
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class EmitterTracker:
    """
    Track super-emitters over time to identify trends, changes, and persistence patterns.
    
    Key features:
    - Temporal linking of detections across time periods
    - Emission trend analysis
    - Change point detection
    - Lifecycle tracking (emergence, persistence, decline)
    - Alert generation for significant changes
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.tracking_params = config['tracking']
        self.persistence_params = self.tracking_params['persistence']
        
        # Tracking state
        self.tracked_emitters = {}  # emitter_id -> tracking data
        self.tracking_history = []
        self.alert_queue = []
        
        logger.info("EmitterTracker initialized")
    
    def track_emitters(self, new_detections: pd.DataFrame, 
                      timestamp: datetime) -> Dict:
        """
        Main tracking function that processes new detections and updates tracks.
        
        Args:
            new_detections: DataFrame with new super-emitter detections
            timestamp: Timestamp for this detection batch
            
        Returns:
            Dictionary with tracking results and alerts
        """
        logger.info(f"Tracking {len(new_detections)} new detections at {timestamp}")
        
        # Step 1: Associate new detections with existing tracks
        associations = self._associate_with_existing_tracks(new_detections, timestamp)
        
        # Step 2: Update existing tracks
        updated_tracks = self._update_existing_tracks(associations, timestamp)
        
        # Step 3: Initialize new tracks for unassociated detections
        new_tracks = self._initialize_new_tracks(associations['unassociated'], timestamp)
        
        # Step 4: Check for missing emitters (potential shutdowns)
        missing_alerts = self._check_missing_emitters(timestamp)
        
        # Step 5: Analyze trends for all active tracks
        trend_results = self._analyze_trends()
        
        # Step 6: Generate alerts for significant changes
        change_alerts = self._detect_significant_changes(trend_results, timestamp)
        
        # Step 7: Update tracking state
        self._update_tracking_state(updated_tracks, new_tracks, timestamp)
        
        # Compile results
        results = {
            'tracking_summary': {
                'timestamp': timestamp,
                'new_detections': len(new_detections),
                'associations_found': len(associations['associated']),
                'new_tracks_started': len(new_tracks),
                'active_tracks': len(self.tracked_emitters),
                'alerts_generated': len(change_alerts) + len(missing_alerts)
            },
            'associations': associations,
            'trend_analysis': trend_results,
            'alerts': change_alerts + missing_alerts,
            'active_emitters': self._get_active_emitters_summary()
        }
        
        logger.info(f"Tracking complete. {len(self.tracked_emitters)} active tracks, "
                   f"{len(results['alerts'])} alerts generated")
        
        return results
    
    def _associate_with_existing_tracks(self, detections: pd.DataFrame, 
                                      timestamp: datetime) -> Dict:
        """Associate new detections with existing emitter tracks."""
        
        if len(detections) == 0:
            return {'associated': [], 'unassociated': detections}
        
        # Get currently active tracks
        active_tracks = self._get_active_tracks(timestamp)
        
        if not active_tracks:
            return {'associated': [], 'unassociated': detections}
        
        # Spatial association parameters
        max_distance_km = 10.0  # Maximum distance for association
        
        associations = []
        associated_indices = []
        
        for idx, detection in detections.iterrows():
            best_match = None
            best_distance = float('inf')
            
            # Check distance to all active tracks
            for track_id, track_data in active_tracks.items():
                # Get latest position from track
                latest_position = track_data['positions'][-1]
                
                # Calculate distance
                distance = self._haversine_distance(
                    detection['center_lat'], detection['center_lon'],
                    latest_position['lat'], latest_position['lon']
                )
                
                if distance < max_distance_km and distance < best_distance:
                    best_distance = distance
                    best_match = track_id
            
            if best_match is not None:
                associations.append({
                    'detection_index': idx,
                    'track_id': best_match,
                    'distance_km': best_distance,
                    'detection_data': detection.to_dict()
                })
                associated_indices.append(idx)
        
        # Separate associated and unassociated detections
        unassociated = detections.drop(associated_indices)
        
        logger.info(f"Associated {len(associations)} detections with existing tracks")
        
        return {
            'associated': associations,
            'unassociated': unassociated
        }
    
    def _get_active_tracks(self, current_time: datetime) -> Dict:
        """Get tracks that are considered active (recent detections)."""
        
        max_gap_days = self.persistence_params['max_gap_days']
        cutoff_time = current_time - timedelta(days=max_gap_days)
        
        active_tracks = {}
        
        for track_id, track_data in self.tracked_emitters.items():
            if track_data['last_detection'] >= cutoff_time:
                active_tracks[track_id] = track_data
        
        return active_tracks
    
    def _update_existing_tracks(self, associations: Dict, timestamp: datetime) -> List[str]:
        """Update existing tracks with new detections."""
        
        updated_track_ids = []
        
        for assoc in associations['associated']:
            track_id = assoc['track_id']
            detection = assoc['detection_data']
            
            # Update track data
            track_data = self.tracked_emitters[track_id]
            
            # Add new position
            track_data['positions'].append({
                'timestamp': timestamp,
                'lat': detection['center_lat'],
                'lon': detection['center_lon'],
                'enhancement': detection['mean_enhancement'],
                'emission_rate': detection['estimated_emission_rate_kg_hr'],
                'detection_score': detection['detection_score']
            })
            
            # Update metadata
            track_data['last_detection'] = timestamp
            track_data['detection_count'] += 1
            track_data['total_detections'] += 1
            
            # Update statistics
            self._update_track_statistics(track_data)
            
            updated_track_ids.append(track_id)
        
        return updated_track_ids
    
    def _initialize_new_tracks(self, unassociated_detections: pd.DataFrame, 
                             timestamp: datetime) -> List[str]:
        """Initialize new tracks for unassociated detections."""
        
        new_track_ids = []
        
        for idx, detection in unassociated_detections.iterrows():
            track_id = f"TRACK_{timestamp.strftime('%Y%m%d')}_{len(self.tracked_emitters):04d}"
            
            # Create new track
            track_data = {
                'track_id': track_id,
                'emitter_id': detection.get('emitter_id', track_id),
                'first_detection': timestamp,
                'last_detection': timestamp,
                'detection_count': 1,
                'total_detections': 1,
                'status': 'active',
                'facility_association': {
                    'facility_id': detection.get('facility_id'),
                    'facility_name': detection.get('facility_name'),
                    'facility_type': detection.get('facility_type')
                },
                'positions': [{
                    'timestamp': timestamp,
                    'lat': detection['center_lat'],
                    'lon': detection['center_lon'],
                    'enhancement': detection['mean_enhancement'],
                    'emission_rate': detection['estimated_emission_rate_kg_hr'],
                    'detection_score': detection['detection_score']
                }],
                'statistics': {
                    'mean_enhancement': detection['mean_enhancement'],
                    'mean_emission_rate': detection['estimated_emission_rate_kg_hr'],
                    'enhancement_trend': None,
                    'emission_trend': None,
                    'last_trend_analysis': None
                },
                'alerts': []
            }
            
            self.tracked_emitters[track_id] = track_data
            new_track_ids.append(track_id)
        
        logger.info(f"Initialized {len(new_track_ids)} new tracks")
        return new_track_ids
    
    def _update_track_statistics(self, track_data: Dict):
        """Update statistical measures for a track."""
        
        positions = track_data['positions']
        
        if len(positions) < 2:
            return
        
        # Extract time series data
        enhancements = [p['enhancement'] for p in positions]
        emission_rates = [p['emission_rate'] for p in positions]
        
        # Update running statistics
        track_data['statistics']['mean_enhancement'] = np.mean(enhancements)
        track_data['statistics']['mean_emission_rate'] = np.mean(emission_rates)
        
        # Calculate trends if enough data points
        if len(positions) >= self.tracking_params['time_series']['min_observations']:
            enhancement_trend = self._calculate_trend(enhancements)
            emission_trend = self._calculate_trend(emission_rates)
            
            track_data['statistics']['enhancement_trend'] = enhancement_trend
            track_data['statistics']['emission_trend'] = emission_trend
            track_data['statistics']['last_trend_analysis'] = datetime.now()
    
    def _calculate_trend(self, values: List[float]) -> Dict:
        """Calculate trend statistics for a time series."""
        
        if len(values) < 3:
            return None
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 3:
            return None
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
        
        # Mann-Kendall trend test
        mk_trend, mk_p_value = self._mann_kendall_test(y_valid)
        
        trend_result = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'mann_kendall_trend': mk_trend,
            'mann_kendall_p_value': mk_p_value,
            'is_significant': p_value < 0.05,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'relative_change_percent': (slope * len(values) / np.mean(y_valid)) * 100 if np.mean(y_valid) > 0 else 0
        }
        
        return trend_result
    
    def _mann_kendall_test(self, data: np.ndarray) -> Tuple[str, float]:
        """Perform Mann-Kendall trend test."""
        
        n = len(data)
        if n < 3:
            return 'insufficient_data', 1.0
        
        # Calculate S statistic
        S = 0
        for i in range(n-1):
            for j in range(i+1, n):
                if data[j] > data[i]:
                    S += 1
                elif data[j] < data[i]:
                    S -= 1
        
        # Calculate variance
        var_S = n * (n - 1) * (2 * n + 5) / 18
        
        if S > 0:
            z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            z = (S + 1) / np.sqrt(var_S)
        else:
            z = 0
        
        # Two-tailed test
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        if abs(z) > 1.96:  # 95% confidence
            trend = 'increasing' if S > 0 else 'decreasing'
        else:
            trend = 'no_trend'
        
        return trend, p_value
    
    def _check_missing_emitters(self, current_time: datetime) -> List[Dict]:
        """Check for emitters that haven't been detected recently (potential shutdowns)."""
        
        max_gap_days = self.persistence_params['max_gap_days']
        alert_threshold_days = max_gap_days * 1.5  # Alert after 1.5x the normal gap
        
        missing_alerts = []
        
        for track_id, track_data in self.tracked_emitters.items():
            if track_data['status'] != 'active':
                continue
            
            days_since_detection = (current_time - track_data['last_detection']).days
            
            if days_since_detection > alert_threshold_days:
                # Check if this is a significant emitter worth alerting about
                mean_emission = track_data['statistics']['mean_emission_rate']
                detection_count = track_data['detection_count']
                
                if mean_emission > 500 and detection_count > 5:  # Significant emitter
                    alert = {
                        'alert_type': 'missing_emitter',
                        'track_id': track_id,
                        'emitter_id': track_data['emitter_id'],
                        'last_detection': track_data['last_detection'],
                        'days_missing': days_since_detection,
                        'mean_emission_rate': mean_emission,
                        'facility_info': track_data['facility_association'],
                        'severity': 'high' if mean_emission > 1000 else 'medium',
                        'message': f"Super-emitter {track_data['emitter_id']} missing for {days_since_detection} days"
                    }
                    missing_alerts.append(alert)
                    
                    # Update track status
                    track_data['status'] = 'missing'
                    track_data['alerts'].append(alert)
        
        return missing_alerts
    
    def _analyze_trends(self) -> Dict:
        """Analyze trends for all active emitters."""
        
        trend_results = {
            'analyzed_tracks': 0,
            'significant_trends': 0,
            'increasing_trends': 0,
            'decreasing_trends': 0,
            'track_details': {}
        }
        
        for track_id, track_data in self.tracked_emitters.items():
            if len(track_data['positions']) < self.tracking_params['time_series']['min_observations']:
                continue
            
            trend_results['analyzed_tracks'] += 1
            
            # Get trend information
            enhancement_trend = track_data['statistics'].get('enhancement_trend')
            emission_trend = track_data['statistics'].get('emission_trend')
            
            if enhancement_trend and enhancement_trend['is_significant']:
                trend_results['significant_trends'] += 1
                
                if enhancement_trend['trend_direction'] == 'increasing':
                    trend_results['increasing_trends'] += 1
                elif enhancement_trend['trend_direction'] == 'decreasing':
                    trend_results['decreasing_trends'] += 1
            
            # Store detailed results
            trend_results['track_details'][track_id] = {
                'enhancement_trend': enhancement_trend,
                'emission_trend': emission_trend,
                'track_duration_days': (track_data['last_detection'] - track_data['first_detection']).days,
                'detection_count': track_data['detection_count']
            }
        
        return trend_results
    
    def _detect_significant_changes(self, trend_results: Dict, 
                                  timestamp: datetime) -> List[Dict]:
        """Detect significant changes that warrant alerts."""
        
        change_alerts = []
        
        for track_id, details in trend_results['track_details'].items():
            track_data = self.tracked_emitters[track_id]
            
            # Check for significant emission increases
            emission_trend = details.get('emission_trend')
            if emission_trend and emission_trend['is_significant']:
                
                change_percent = abs(emission_trend['relative_change_percent'])
                threshold_percent = self.config['alerts']['thresholds']['emission_increase_percent']
                
                if change_percent > threshold_percent:
                    alert_type = 'emission_increase' if emission_trend['slope'] > 0 else 'emission_decrease'
                    
                    # Determine severity
                    if change_percent > threshold_percent * 2:
                        severity = 'high'
                    elif change_percent > threshold_percent * 1.5:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    alert = {
                        'alert_type': alert_type,
                        'track_id': track_id,
                        'emitter_id': track_data['emitter_id'],
                        'timestamp': timestamp,
                        'change_percent': change_percent,
                        'trend_direction': emission_trend['trend_direction'],
                        'p_value': emission_trend['p_value'],
                        'current_emission_rate': track_data['positions'][-1]['emission_rate'],
                        'mean_emission_rate': track_data['statistics']['mean_emission_rate'],
                        'facility_info': track_data['facility_association'],
                        'severity': severity,
                        'message': f"Significant {alert_type.replace('_', ' ')} detected: {change_percent:.1f}% change"
                    }
                    
                    change_alerts.append(alert)
                    track_data['alerts'].append(alert)
        
        # Check for new super-emitters (tracks that just became significant)
        new_emitter_alerts = self._detect_new_super_emitters(timestamp)
        change_alerts.extend(new_emitter_alerts)
        
        return change_alerts
    
    def _detect_new_super_emitters(self, timestamp: datetime) -> List[Dict]:
        """Detect newly emerged super-emitters."""
        
        new_emitter_alerts = []
        threshold = self.config['super_emitters']['detection']['emission_rate_threshold']
        confidence_threshold = self.config['alerts']['thresholds']['new_emitter_confidence']
        
        for track_id, track_data in self.tracked_emitters.items():
            # Check for tracks that recently crossed the super-emitter threshold
            if track_data['detection_count'] >= 3:  # Need some confidence
                
                recent_emissions = [p['emission_rate'] for p in track_data['positions'][-3:]]
                mean_recent = np.mean(recent_emissions)
                detection_score = np.mean([p['detection_score'] for p in track_data['positions'][-3:]])
                
                # Check if this is a new super-emitter
                track_age_days = (timestamp - track_data['first_detection']).days
                
                if (mean_recent > threshold and 
                    detection_score > confidence_threshold and 
                    track_age_days <= 7 and  # Recently detected
                    not any(alert['alert_type'] == 'new_super_emitter' for alert in track_data['alerts'])):
                    
                    alert = {
                        'alert_type': 'new_super_emitter',
                        'track_id': track_id,
                        'emitter_id': track_data['emitter_id'],
                        'timestamp': timestamp,
                        'emission_rate': mean_recent,
                        'detection_score': detection_score,
                        'track_age_days': track_age_days,
                        'facility_info': track_data['facility_association'],
                        'severity': 'high' if mean_recent > threshold * 2 else 'medium',
                        'message': f"New super-emitter detected: {mean_recent:.0f} kg/hr"
                    }
                    
                    new_emitter_alerts.append(alert)
                    track_data['alerts'].append(alert)
        
        return new_emitter_alerts
    
    def _update_tracking_state(self, updated_tracks: List[str], 
                             new_tracks: List[str], timestamp: datetime):
        """Update the overall tracking state and cleanup old tracks."""
        
        # Update tracking history
        self.tracking_history.append({
            'timestamp': timestamp,
            'active_tracks': len(self.tracked_emitters),
            'updated_tracks': len(updated_tracks),
            'new_tracks': len(new_tracks)
        })
        
        # Cleanup old tracking history (keep last 30 days)
        cutoff_time = timestamp - timedelta(days=30)
        self.tracking_history = [
            h for h in self.tracking_history 
            if h['timestamp'] >= cutoff_time
        ]
        
        # Archive very old tracks that haven't been seen in a long time
        self._archive_old_tracks(timestamp)
    
    def _archive_old_tracks(self, current_time: datetime):
        """Archive tracks that haven't been active for a long time."""
        
        archive_threshold_days = 90  # Archive after 90 days of inactivity
        cutoff_time = current_time - timedelta(days=archive_threshold_days)
        
        tracks_to_archive = []
        
        for track_id, track_data in self.tracked_emitters.items():
            if track_data['last_detection'] < cutoff_time:
                tracks_to_archive.append(track_id)
                track_data['status'] = 'archived'
        
        # Remove from active tracking
        for track_id in tracks_to_archive:
            archived_track = self.tracked_emitters.pop(track_id)
            logger.info(f"Archived inactive track: {track_id}")
        
        if tracks_to_archive:
            logger.info(f"Archived {len(tracks_to_archive)} inactive tracks")
    
    def _get_active_emitters_summary(self) -> Dict:
        """Get summary statistics for currently active emitters."""
        
        if not self.tracked_emitters:
            return {'total_active': 0}
        
        active_tracks = [t for t in self.tracked_emitters.values() if t['status'] == 'active']
        
        if not active_tracks:
            return {'total_active': 0}
        
        # Calculate summary statistics
        emission_rates = [t['statistics']['mean_emission_rate'] for t in active_tracks]
        detection_counts = [t['detection_count'] for t in active_tracks]
        
        # Facility associations
        associated_tracks = [t for t in active_tracks if t['facility_association']['facility_id'] is not None]
        
        summary = {
            'total_active': len(active_tracks),
            'total_emission_rate_kg_hr': sum(emission_rates),
            'mean_emission_rate_kg_hr': np.mean(emission_rates),
            'max_emission_rate_kg_hr': max(emission_rates),
            'facility_associated': len(associated_tracks),
            'facility_association_rate': len(associated_tracks) / len(active_tracks),
            'mean_detections_per_track': np.mean(detection_counts),
            'tracks_with_trends': len([t for t in active_tracks 
                                     if t['statistics'].get('enhancement_trend') and 
                                     t['statistics']['enhancement_trend']['is_significant']])
        }
        
        return summary
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points in km."""
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of earth in km
        r = 6371
        
        return c * r
    
    def get_track_details(self, track_id: str) -> Optional[Dict]:
        """Get detailed information about a specific track."""
        
        if track_id not in self.tracked_emitters:
            return None
        
        track_data = self.tracked_emitters[track_id].copy()
        
        # Add computed fields
        positions = track_data['positions']
        if len(positions) > 1:
            # Calculate movement statistics
            distances = []
            for i in range(1, len(positions)):
                dist = self._haversine_distance(
                    positions[i-1]['lat'], positions[i-1]['lon'],
                    positions[i]['lat'], positions[i]['lon']
                )
                distances.append(dist)
            
            track_data['movement_stats'] = {
                'total_movement_km': sum(distances),
                'mean_movement_km': np.mean(distances),
                'max_movement_km': max(distances),
                'is_stationary': all(d < 1.0 for d in distances)  # Less than 1km movement
            }
        
        return track_data
    
    def get_tracking_summary(self) -> Dict:
        """Get overall tracking system summary."""
        
        active_count = len([t for t in self.tracked_emitters.values() if t['status'] == 'active'])
        missing_count = len([t for t in self.tracked_emitters.values() if t['status'] == 'missing'])
        
        # Recent activity (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_detections = sum(
            1 for t in self.tracked_emitters.values() 
            if t['last_detection'] >= week_ago
        )
        
        summary = {
            'total_tracks': len(self.tracked_emitters),
            'active_tracks': active_count,
            'missing_tracks': missing_count,
            'recent_activity_7days': recent_detections,
            'total_alerts': len(self.alert_queue),
            'tracking_history_length': len(self.tracking_history)
        }
        
        return summary
    
    def export_tracks_to_dataframe(self) -> pd.DataFrame:
        """Export all track data to a pandas DataFrame for analysis."""
        
        if not self.tracked_emitters:
            return pd.DataFrame()
        
        track_records = []
        
        for track_id, track_data in self.tracked_emitters.items():
            for position in track_data['positions']:
                record = {
                    'track_id': track_id,
                    'emitter_id': track_data['emitter_id'],
                    'timestamp': position['timestamp'],
                    'lat': position['lat'],
                    'lon': position['lon'],
                    'enhancement': position['enhancement'],
                    'emission_rate': position['emission_rate'],
                    'detection_score': position['detection_score'],
                    'status': track_data['status'],
                    'facility_id': track_data['facility_association']['facility_id'],
                    'facility_name': track_data['facility_association']['facility_name'],
                    'facility_type': track_data['facility_association']['facility_type'],
                    'track_age_days': (position['timestamp'] - track_data['first_detection']).days,
                    'total_detections': track_data['total_detections']
                }
                
                # Add trend information if available
                if track_data['statistics'].get('enhancement_trend'):
                    trend = track_data['statistics']['enhancement_trend']
                    record.update({
                        'trend_slope': trend['slope'],
                        'trend_r_squared': trend['r_squared'],
                        'trend_p_value': trend['p_value'],
                        'trend_direction': trend['trend_direction'],
                        'trend_significant': trend['is_significant']
                    })
                
                track_records.append(record)
        
        return pd.DataFrame(track_records)