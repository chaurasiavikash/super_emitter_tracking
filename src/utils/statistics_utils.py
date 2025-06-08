# ============================================================================
# FILE: src/utils/validation_utils.py
# ============================================================================
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import cdist
from scipy import stats
import logging
from typing import Dict, List, Tuple, Optional, Union
from shapely.geometry import Point
import warnings

logger = logging.getLogger(__name__)

def validate_against_reference(detections: pd.DataFrame, 
                             reference_data: pd.DataFrame,
                             spatial_tolerance_km: float = 10.0,
                             temporal_tolerance_hours: float = 24.0,
                             emission_tolerance_percent: float = 50.0) -> Dict:
    """
    Validate super-emitter detections against reference/ground truth data.
    
    Args:
        detections: DataFrame with detected super-emitters
        reference_data: DataFrame with reference/ground truth data
        spatial_tolerance_km: Maximum distance for spatial matching (km)
        temporal_tolerance_hours: Maximum time difference for temporal matching (hours)
        emission_tolerance_percent: Tolerance for emission rate comparison (%)
        
    Returns:
        Dictionary with validation metrics and matched pairs
    """
    logger.info("Starting validation against reference data")
    
    if len(detections) == 0:
        logger.warning("No detections to validate")
        return _create_empty_validation_results()
    
    if len(reference_data) == 0:
        logger.warning("No reference data available for validation")
        return _create_empty_validation_results()
    
    # Ensure required columns exist
    required_detection_cols = ['center_lat', 'center_lon', 'estimated_emission_rate_kg_hr']
    required_reference_cols = ['lat', 'lon', 'emission_rate_kg_hr']
    
    missing_det_cols = [col for col in required_detection_cols if col not in detections.columns]
    missing_ref_cols = [col for col in required_reference_cols if col not in reference_data.columns]
    
    if missing_det_cols:
        logger.error(f"Missing columns in detections: {missing_det_cols}")
        return _create_empty_validation_results()
    
    if missing_ref_cols:
        logger.error(f"Missing columns in reference data: {missing_ref_cols}")
        return _create_empty_validation_results()
    
    # Find spatial matches
    spatial_matches = _find_spatial_matches(
        detections, reference_data, spatial_tolerance_km
    )
    
    # Apply temporal filtering if timestamp data available
    if 'timestamp' in detections.columns and 'timestamp' in reference_data.columns:
        temporal_matches = _apply_temporal_filter(
            spatial_matches, temporal_tolerance_hours
        )
    else:
        temporal_matches = spatial_matches
    
    # Calculate validation metrics
    validation_metrics = _calculate_validation_metrics(
        detections, reference_data, temporal_matches
    )
    
    # Analyze emission rate accuracy
    emission_analysis = _analyze_emission_accuracy(
        temporal_matches, emission_tolerance_percent
    )
    
    # Spatial distribution analysis
    spatial_analysis = _analyze_spatial_distribution(
        detections, reference_data, temporal_matches
    )
    
    # Compile results
    validation_results = {
        'validation_summary': {
            'total_detections': len(detections),
            'total_reference': len(reference_data),
            'spatial_matches': len(spatial_matches),
            'temporal_matches': len(temporal_matches),
            'match_success_rate': len(temporal_matches) / len(detections) if len(detections) > 0 else 0
        },
        'detection_metrics': validation_metrics,
        'emission_accuracy': emission_analysis,
        'spatial_analysis': spatial_analysis,
        'matched_pairs': temporal_matches,
        'validation_parameters': {
            'spatial_tolerance_km': spatial_tolerance_km,
            'temporal_tolerance_hours': temporal_tolerance_hours,
            'emission_tolerance_percent': emission_tolerance_percent
        }
    }
    
    logger.info(f"Validation complete: {len(temporal_matches)} matches found")
    return validation_results

def _find_spatial_matches(detections: pd.DataFrame, 
                         reference_data: pd.DataFrame,
                         tolerance_km: float) -> List[Dict]:
    """Find spatial matches between detections and reference data."""
    
    # Convert DataFrames to GeoDataFrames
    det_points = [Point(lon, lat) for lat, lon in 
                  zip(detections['center_lat'], detections['center_lon'])]
    det_gdf = gpd.GeoDataFrame(detections, geometry=det_points, crs='EPSG:4326')
    
    ref_points = [Point(lon, lat) for lat, lon in 
                  zip(reference_data['lat'], reference_data['lon'])]
    ref_gdf = gpd.GeoDataFrame(reference_data, geometry=ref_points, crs='EPSG:4326')
    
    # Convert to projected CRS for distance calculations
    # Use appropriate UTM zone or equal area projection
    det_gdf_proj = det_gdf.to_crs('EPSG:3857')  # Web Mercator for global coverage
    ref_gdf_proj = ref_gdf.to_crs('EPSG:3857')
    
    matches = []
    tolerance_m = tolerance_km * 1000  # Convert to meters
    
    for det_idx, det_row in det_gdf_proj.iterrows():
        # Calculate distances to all reference points
        distances = ref_gdf_proj.geometry.distance(det_row.geometry)
        
        # Find reference points within tolerance
        close_refs = distances[distances <= tolerance_m]
        
        if len(close_refs) > 0:
            # Find closest reference point
            closest_ref_idx = distances.idxmin()
            distance_km = distances[closest_ref_idx] / 1000
            
            match = {
                'detection_index': det_idx,
                'reference_index': closest_ref_idx,
                'distance_km': distance_km,
                'detection_data': detections.iloc[det_idx].to_dict(),
                'reference_data': reference_data.iloc[closest_ref_idx].to_dict()
            }
            matches.append(match)
    
    return matches

def _apply_temporal_filter(spatial_matches: List[Dict], 
                          tolerance_hours: float) -> List[Dict]:
    """Apply temporal filtering to spatial matches."""
    
    temporal_matches = []
    
    for match in spatial_matches:
        det_time = pd.to_datetime(match['detection_data'].get('timestamp'))
        ref_time = pd.to_datetime(match['reference_data'].get('timestamp'))
        
        if pd.isna(det_time) or pd.isna(ref_time):
            # If no temporal info, keep the match
            temporal_matches.append(match)
            continue
        
        time_diff_hours = abs((det_time - ref_time).total_seconds()) / 3600
        
        if time_diff_hours <= tolerance_hours:
            match['temporal_difference_hours'] = time_diff_hours
            temporal_matches.append(match)
    
    return temporal_matches

def _calculate_validation_metrics(detections: pd.DataFrame,
                                reference_data: pd.DataFrame,
                                matches: List[Dict]) -> Dict:
    """Calculate standard validation metrics (precision, recall, F1)."""
    
    n_detections = len(detections)
    n_reference = len(reference_data)
    n_matches = len(matches)
    
    # True Positives: Detections that match reference data
    true_positives = n_matches
    
    # False Positives: Detections that don't match any reference data
    false_positives = n_detections - n_matches
    
    # False Negatives: Reference data that wasn't detected
    matched_ref_indices = [match['reference_index'] for match in matches]
    false_negatives = n_reference - len(set(matched_ref_indices))
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Additional metrics
    accuracy = true_positives / max(n_detections, n_reference) if max(n_detections, n_reference) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'detection_rate': n_matches / n_reference if n_reference > 0 else 0,
        'false_discovery_rate': false_positives / n_detections if n_detections > 0 else 0
    }

def _analyze_emission_accuracy(matches: List[Dict], 
                              tolerance_percent: float) -> Dict:
    """Analyze accuracy of emission rate estimates."""
    
    if not matches:
        return {
            'emission_correlation': 0,
            'emission_rmse': 0,
            'emission_mae': 0,
            'emission_bias': 0,
            'within_tolerance_fraction': 0,
            'emission_r2': 0
        }
    
    detected_emissions = []
    reference_emissions = []
    
    for match in matches:
        det_emission = match['detection_data']['estimated_emission_rate_kg_hr']
        ref_emission = match['reference_data']['emission_rate_kg_hr']
        
        if not (pd.isna(det_emission) or pd.isna(ref_emission)):
            detected_emissions.append(det_emission)
            reference_emissions.append(ref_emission)
    
    if len(detected_emissions) < 2:
        return {
            'emission_correlation': 0,
            'emission_rmse': 0,
            'emission_mae': 0,
            'emission_bias': 0,
            'within_tolerance_fraction': 0,
            'emission_r2': 0
        }
    
    detected_emissions = np.array(detected_emissions)
    reference_emissions = np.array(reference_emissions)
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(detected_emissions, reference_emissions)
    
    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(reference_emissions, detected_emissions))
    mae = mean_absolute_error(reference_emissions, reference_emissions)
    bias = np.mean(detected_emissions - reference_emissions)
    r2 = r2_score(reference_emissions, detected_emissions)
    
    # Calculate fraction within tolerance
    relative_errors = np.abs(detected_emissions - reference_emissions) / reference_emissions * 100
    within_tolerance = np.sum(relative_errors <= tolerance_percent) / len(relative_errors)
    
    return {
        'emission_correlation': correlation,
        'emission_correlation_p_value': p_value,
        'emission_rmse': rmse,
        'emission_mae': mae,
        'emission_bias': bias,
        'emission_r2': r2,
        'within_tolerance_fraction': within_tolerance,
        'mean_relative_error_percent': np.mean(relative_errors),
        'median_relative_error_percent': np.median(relative_errors),
        'n_comparisons': len(detected_emissions)
    }

def _analyze_spatial_distribution(detections: pd.DataFrame,
                                reference_data: pd.DataFrame,
                                matches: List[Dict]) -> Dict:
    """Analyze spatial distribution of detections vs reference."""
    
    # Calculate spatial coverage
    det_lat_range = [detections['center_lat'].min(), detections['center_lat'].max()]
    det_lon_range = [detections['center_lon'].min(), detections['center_lon'].max()]
    
    ref_lat_range = [reference_data['lat'].min(), reference_data['lat'].max()]
    ref_lon_range = [reference_data['lon'].min(), reference_data['lon'].max()]
    
    # Calculate overlap
    lat_overlap = max(0, min(det_lat_range[1], ref_lat_range[1]) - max(det_lat_range[0], ref_lat_range[0]))
    lon_overlap = max(0, min(det_lon_range[1], ref_lon_range[1]) - max(det_lon_range[0], ref_lon_range[0]))
    
    det_lat_span = det_lat_range[1] - det_lat_range[0]
    det_lon_span = det_lon_range[1] - det_lon_range[0]
    
    spatial_overlap_fraction = (lat_overlap * lon_overlap) / (det_lat_span * det_lon_span) if (det_lat_span * det_lon_span) > 0 else 0
    
    # Distance statistics for matches
    if matches:
        distances = [match['distance_km'] for match in matches]
        distance_stats = {
            'mean_distance_km': np.mean(distances),
            'median_distance_km': np.median(distances),
            'max_distance_km': np.max(distances),
            'std_distance_km': np.std(distances)
        }
    else:
        distance_stats = {
            'mean_distance_km': 0,
            'median_distance_km': 0,
            'max_distance_km': 0,
            'std_distance_km': 0
        }
    
    return {
        'detection_spatial_range': {
            'lat_range': det_lat_range,
            'lon_range': det_lon_range
        },
        'reference_spatial_range': {
            'lat_range': ref_lat_range,
            'lon_range': ref_lon_range
        },
        'spatial_overlap_fraction': spatial_overlap_fraction,
        'distance_statistics': distance_stats
    }

def _create_empty_validation_results() -> Dict:
    """Create empty validation results structure."""
    
    return {
        'validation_summary': {
            'total_detections': 0,
            'total_reference': 0,
            'spatial_matches': 0,
            'temporal_matches': 0,
            'match_success_rate': 0
        },
        'detection_metrics': {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'accuracy': 0
        },
        'emission_accuracy': {
            'emission_correlation': 0,
            'emission_rmse': 0,
            'within_tolerance_fraction': 0
        },
        'spatial_analysis': {},
        'matched_pairs': [],
        'validation_parameters': {}
    }

def calculate_detection_performance(detections: pd.DataFrame,
                                  ground_truth: pd.DataFrame,
                                  spatial_threshold_km: float = 5.0) -> Dict:
    """
    Calculate comprehensive detection performance metrics.
    
    Args:
        detections: Detected super-emitters
        ground_truth: Known super-emitters (ground truth)
        spatial_threshold_km: Distance threshold for matching
        
    Returns:
        Performance metrics dictionary
    """
    
    logger.info("Calculating detection performance metrics")
    
    if len(detections) == 0 and len(ground_truth) == 0:
        return {'no_data': True}
    
    # Create binary classification arrays
    # 1 = emitter present, 0 = no emitter
    
    # For simplicity, create a grid and mark presence/absence
    # In practice, you might use actual facility locations
    
    # Get spatial bounds
    all_lats = []
    all_lons = []
    
    if len(detections) > 0:
        all_lats.extend(detections['center_lat'].tolist())
        all_lons.extend(detections['center_lon'].tolist())
    
    if len(ground_truth) > 0:
        all_lats.extend(ground_truth['lat'].tolist())
        all_lons.extend(ground_truth['lon'].tolist())
    
    if not all_lats:
        return {'no_spatial_data': True}
    
    # Create spatial grid
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)
    
    # Grid resolution based on spatial threshold
    grid_res = spatial_threshold_km / 111.32  # Convert km to degrees (approximate)
    
    lat_grid = np.arange(lat_min, lat_max + grid_res, grid_res)
    lon_grid = np.arange(lon_min, lon_max + grid_res, grid_res)
    
    # Mark grid cells with detections and ground truth
    detected_grid = np.zeros((len(lat_grid), len(lon_grid)))
    truth_grid = np.zeros((len(lat_grid), len(lon_grid)))
    
    # Mark detected emitters
    for _, det in detections.iterrows():
        lat_idx = np.argmin(np.abs(lat_grid - det['center_lat']))
        lon_idx = np.argmin(np.abs(lon_grid - det['center_lon']))
        detected_grid[lat_idx, lon_idx] = 1
    
    # Mark ground truth emitters
    for _, truth in ground_truth.iterrows():
        lat_idx = np.argmin(np.abs(lat_grid - truth['lat']))
        lon_idx = np.argmin(np.abs(lon_grid - truth['lon']))
        truth_grid[lat_idx, lon_idx] = 1
    
    # Calculate confusion matrix elements
    tp = np.sum((detected_grid == 1) & (truth_grid == 1))
    fp = np.sum((detected_grid == 1) & (truth_grid == 0))
    fn = np.sum((detected_grid == 0) & (truth_grid == 1))
    tn = np.sum((detected_grid == 0) & (truth_grid == 0))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'confusion_matrix': {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        },
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'specificity': specificity
        },
        'grid_info': {
            'grid_resolution_deg': grid_res,
            'grid_size': detected_grid.shape,
            'total_cells': detected_grid.size
        }
    }

def validate_emission_quantification(estimated_emissions: np.ndarray,
                                   reference_emissions: np.ndarray,
                                   uncertainty_estimates: Optional[np.ndarray] = None) -> Dict:
    """
    Validate emission rate quantification accuracy.
    
    Args:
        estimated_emissions: Estimated emission rates
        reference_emissions: Reference/true emission rates
        uncertainty_estimates: Optional uncertainty estimates
        
    Returns:
        Quantification validation metrics
    """
    
    logger.info("Validating emission rate quantification")
    
    # Remove NaN values
    valid_mask = ~(np.isnan(estimated_emissions) | np.isnan(reference_emissions))
    
    if np.sum(valid_mask) < 2:
        logger.warning("Insufficient valid data for quantification validation")
        return {'insufficient_data': True}
    
    est_valid = estimated_emissions[valid_mask]
    ref_valid = reference_emissions[valid_mask]
    
    # Basic statistics
    correlation, p_value = stats.pearsonr(est_valid, ref_valid)
    r2 = r2_score(ref_valid, est_valid)
    rmse = np.sqrt(mean_squared_error(ref_valid, est_valid))
    mae = mean_absolute_error(ref_valid, est_valid)
    
    # Bias analysis
    bias = np.mean(est_valid - ref_valid)
    relative_bias = bias / np.mean(ref_valid) * 100
    
    # Error distribution
    errors = est_valid - ref_valid
    relative_errors = errors / ref_valid * 100
    
    # Performance by emission magnitude
    low_emissions = ref_valid <= np.percentile(ref_valid, 33)
    med_emissions = (ref_valid > np.percentile(ref_valid, 33)) & (ref_valid <= np.percentile(ref_valid, 67))
    high_emissions = ref_valid > np.percentile(ref_valid, 67)
    
    magnitude_performance = {}
    for magnitude, mask in [('low', low_emissions), ('medium', med_emissions), ('high', high_emissions)]:
        if np.sum(mask) > 1:
            magnitude_performance[magnitude] = {
                'r2': r2_score(ref_valid[mask], est_valid[mask]),
                'rmse': np.sqrt(mean_squared_error(ref_valid[mask], est_valid[mask])),
                'bias': np.mean(est_valid[mask] - ref_valid[mask]),
                'n_samples': np.sum(mask)
            }
    
    results = {
        'overall_performance': {
            'correlation': correlation,
            'correlation_p_value': p_value,
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'relative_bias_percent': relative_bias
        },
        'error_statistics': {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'mean_absolute_error': np.mean(np.abs(errors)),
            'mean_relative_error_percent': np.mean(np.abs(relative_errors)),
            'median_relative_error_percent': np.median(np.abs(relative_errors))
        },
        'performance_by_magnitude': magnitude_performance,
        'sample_info': {
            'n_valid_pairs': len(est_valid),
            'reference_range': [float(np.min(ref_valid)), float(np.max(ref_valid))],
            'estimated_range': [float(np.min(est_valid)), float(np.max(est_valid))]
        }
    }
    
    # Uncertainty analysis if provided
    if uncertainty_estimates is not None:
        unc_valid = uncertainty_estimates[valid_mask]
        results['uncertainty_analysis'] = _analyze_uncertainty_estimates(
            est_valid, ref_valid, unc_valid
        )
    
    return results

def _analyze_uncertainty_estimates(estimated: np.ndarray,
                                 reference: np.ndarray,
                                 uncertainties: np.ndarray) -> Dict:
    """Analyze quality of uncertainty estimates."""
    
    errors = np.abs(estimated - reference)
    
    # Check if errors are within uncertainty bounds
    within_1sigma = errors <= uncertainties
    within_2sigma = errors <= (2 * uncertainties)
    
    # Coverage probability
    coverage_1sigma = np.mean(within_1sigma)
    coverage_2sigma = np.mean(within_2sigma)
    
    # Reliability of uncertainty estimates
    # Good uncertainty estimates should have ~68% coverage at 1-sigma
    reliability_1sigma = abs(coverage_1sigma - 0.68)
    reliability_2sigma = abs(coverage_2sigma - 0.95)
    
    return {
        'coverage_1sigma': coverage_1sigma,
        'coverage_2sigma': coverage_2sigma,
        'reliability_1sigma': reliability_1sigma,
        'reliability_2sigma': reliability_2sigma,
        'mean_uncertainty': np.mean(uncertainties),
        'uncertainty_vs_error_correlation': stats.pearsonr(uncertainties, errors)[0]
    }

def cross_validate_detection_algorithm(detection_function,
                                     dataset: pd.DataFrame,
                                     n_folds: int = 5,
                                     spatial_cv: bool = True) -> Dict:
    """
    Perform cross-validation of detection algorithm.
    
    Args:
        detection_function: Function that takes dataset and returns detections
        dataset: Full dataset for cross-validation
        n_folds: Number of CV folds
        spatial_cv: Whether to use spatial cross-validation
        
    Returns:
        Cross-validation results
    """
    
    logger.info(f"Performing {n_folds}-fold cross-validation")
    
    if spatial_cv:
        # Spatial cross-validation - split by geographic regions
        folds = _create_spatial_folds(dataset, n_folds)
    else:
        # Random cross-validation
        folds = _create_random_folds(dataset, n_folds)
    
    cv_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        logger.info(f"Processing fold {fold_idx + 1}/{n_folds}")
        
        train_data = dataset.iloc[train_idx]
        test_data = dataset.iloc[test_idx]
        
        try:
            # Train/fit on training data and predict on test data
            detections = detection_function(train_data, test_data)
            
            # Validate against test data
            if 'ground_truth' in test_data.columns:
                ground_truth = test_data[test_data['ground_truth'] == True]
                validation_results = validate_against_reference(
                    detections, ground_truth
                )
                
                cv_results.append({
                    'fold': fold_idx,
                    'n_train': len(train_data),
                    'n_test': len(test_data),
                    'n_detections': len(detections),
                    'validation_metrics': validation_results['detection_metrics']
                })
            
        except Exception as e:
            logger.warning(f"Fold {fold_idx} failed: {e}")
            continue
    
    # Aggregate results
    if cv_results:
        avg_precision = np.mean([r['validation_metrics']['precision'] for r in cv_results])
        avg_recall = np.mean([r['validation_metrics']['recall'] for r in cv_results])
        avg_f1 = np.mean([r['validation_metrics']['f1_score'] for r in cv_results])
        
        std_precision = np.std([r['validation_metrics']['precision'] for r in cv_results])
        std_recall = np.std([r['validation_metrics']['recall'] for r in cv_results])
        std_f1 = np.std([r['validation_metrics']['f1_score'] for r in cv_results])
        
        return {
            'cross_validation_summary': {
                'n_folds': len(cv_results),
                'spatial_cv': spatial_cv,
                'mean_precision': avg_precision,
                'std_precision': std_precision,
                'mean_recall': avg_recall,
                'std_recall': std_recall,
                'mean_f1_score': avg_f1,
                'std_f1_score': std_f1
            },
            'fold_results': cv_results
        }
    else:
        return {'cross_validation_failed': True}

def _create_spatial_folds(dataset: pd.DataFrame, n_folds: int) -> List[Tuple]:
    """Create spatial cross-validation folds."""
    
    if 'lat' not in dataset.columns or 'lon' not in dataset.columns:
        logger.warning("No spatial coordinates for spatial CV, using random folds")
        return _create_random_folds(dataset, n_folds)
    
    # Simple spatial folding: divide by latitude bands
    lat_min, lat_max = dataset['lat'].min(), dataset['lat'].max()
    lat_bands = np.linspace(lat_min, lat_max, n_folds + 1)
    
    folds = []
    for i in range(n_folds):
        # Test fold: current latitude band
        test_mask = (dataset['lat'] >= lat_bands[i]) & (dataset['lat'] < lat_bands[i + 1])
        if i == n_folds - 1:  # Include upper bound for last fold
            test_mask = (dataset['lat'] >= lat_bands[i]) & (dataset['lat'] <= lat_bands[i + 1])
        
        test_idx = dataset.index[test_mask].tolist()
        train_idx = dataset.index[~test_mask].tolist()
        
        folds.append((train_idx, test_idx))
    
    return folds

def _create_random_folds(dataset: pd.DataFrame, n_folds: int) -> List[Tuple]:
    """Create random cross-validation folds."""
    
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(kf.split(dataset))
    
    return folds