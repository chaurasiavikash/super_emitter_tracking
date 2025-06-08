# ============================================================================
# FILE: src/utils/geo_utils.py
# ============================================================================
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Optional
import logging
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import warnings

logger = logging.getLogger(__name__)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
        
    Returns:
        Distance in kilometers
    """
    
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r

def calculate_pixel_area(lat: float, lon: float, 
                        pixel_size_lat: float = 0.01, 
                        pixel_size_lon: float = 0.01) -> float:
    """
    Calculate the area of a pixel in km² at given coordinates.
    
    Args:
        lat: Latitude (degrees)
        lon: Longitude (degrees)
        pixel_size_lat: Pixel size in latitude direction (degrees)
        pixel_size_lon: Pixel size in longitude direction (degrees)
        
    Returns:
        Pixel area in km²
    """
    
    # Earth radius in km
    R = 6371.0
    
    # Convert to radians
    lat_rad = radians(lat)
    
    # Calculate distances
    lat_distance = pixel_size_lat * (np.pi * R / 180.0)
    lon_distance = pixel_size_lon * (np.pi * R / 180.0) * cos(lat_rad)
    
    # Area in km²
    area = lat_distance * lon_distance
    
    return area

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing between two points.
    
    Args:
        lat1, lon1: Starting point coordinates (degrees)
        lat2, lon2: Ending point coordinates (degrees)
        
    Returns:
        Bearing in degrees (0-360, where 0 is north)
    """
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    
    bearing = atan2(y, x)
    
    # Convert to degrees and normalize to 0-360
    bearing = degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing

def project_coordinates(lats: np.ndarray, lons: np.ndarray, 
                       projection: str = 'mercator') -> Tuple[np.ndarray, np.ndarray]:
    """
    Project geographic coordinates to Cartesian coordinates.
    
    Args:
        lats: Array of latitudes (degrees)
        lons: Array of longitudes (degrees)
        projection: Projection type ('mercator', 'equirectangular', 'stereographic')
        
    Returns:
        Tuple of (x, y) coordinates in meters
    """
    
    # Earth radius in meters
    R = 6378137.0  # WGS84 equatorial radius
    
    if projection == 'mercator':
        # Web Mercator projection
        x = lons * np.pi * R / 180.0
        y = R * np.log(np.tan(np.pi/4 + lats * np.pi / 360.0))
        
    elif projection == 'equirectangular':
        # Simple equirectangular projection
        lat_center = np.mean(lats)
        x = lons * np.pi * R * np.cos(np.radians(lat_center)) / 180.0
        y = lats * np.pi * R / 180.0
        
    elif projection == 'stereographic':
        # Polar stereographic (simplified)
        lat_center = np.mean(lats)
        lon_center = np.mean(lons)
        
        # Convert to radians
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)
        lat_c = np.radians(lat_center)
        lon_c = np.radians(lon_center)
        
        # Stereographic projection
        k = 2 * R / (1 + np.sin(lat_c) * np.sin(lat_rad) + 
                     np.cos(lat_c) * np.cos(lat_rad) * np.cos(lon_rad - lon_c))
        
        x = k * np.cos(lat_rad) * np.sin(lon_rad - lon_c)
        y = k * (np.cos(lat_c) * np.sin(lat_rad) - 
                 np.sin(lat_c) * np.cos(lat_rad) * np.cos(lon_rad - lon_c))
        
    else:
        raise ValueError(f"Unknown projection: {projection}")
    
    return x, y

def calculate_grid_spacing(lat_min: float, lat_max: float, 
                          lon_min: float, lon_max: float,
                          target_resolution_km: float) -> Tuple[float, float]:
    """
    Calculate grid spacing in degrees for target resolution in km.
    
    Args:
        lat_min, lat_max: Latitude bounds (degrees)
        lon_min, lon_max: Longitude bounds (degrees)
        target_resolution_km: Target resolution in kilometers
        
    Returns:
        Tuple of (lat_spacing, lon_spacing) in degrees
    """
    
    # Average latitude for longitude correction
    lat_center = (lat_min + lat_max) / 2
    
    # Approximate conversion factors at center latitude
    deg_to_km_lat = 111.32  # km per degree latitude
    deg_to_km_lon = 111.32 * cos(radians(lat_center))  # km per degree longitude
    
    # Calculate spacing in degrees
    lat_spacing = target_resolution_km / deg_to_km_lat
    lon_spacing = target_resolution_km / deg_to_km_lon
    
    return lat_spacing, lon_spacing

def create_regular_grid(lat_min: float, lat_max: float,
                       lon_min: float, lon_max: float,
                       resolution_km: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a regular geographic grid.
    
    Args:
        lat_min, lat_max: Latitude bounds (degrees)
        lon_min, lon_max: Longitude bounds (degrees)
        resolution_km: Grid resolution in kilometers
        
    Returns:
        Tuple of (latitude_grid, longitude_grid) 2D arrays
    """
    
    # Calculate grid spacing
    lat_spacing, lon_spacing = calculate_grid_spacing(
        lat_min, lat_max, lon_min, lon_max, resolution_km
    )
    
    # Create 1D arrays
    lats = np.arange(lat_min, lat_max + lat_spacing, lat_spacing)
    lons = np.arange(lon_min, lon_max + lon_spacing, lon_spacing)
    
    # Create 2D grids
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    return lat_grid, lon_grid

def calculate_area_from_coordinates(lats: np.ndarray, lons: np.ndarray) -> float:
    """
    Calculate area of a polygon defined by coordinates using spherical excess.
    
    Args:
        lats: Array of latitude coordinates (degrees)
        lons: Array of longitude coordinates (degrees)
        
    Returns:
        Area in km²
    """
    
    if len(lats) < 3:
        return 0.0
    
    # Ensure polygon is closed
    if lats[0] != lats[-1] or lons[0] != lons[-1]:
        lats = np.append(lats, lats[0])
        lons = np.append(lons, lons[0])
    
    # Convert to radians
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    
    # Calculate area using spherical excess formula
    n = len(lats_rad) - 1
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += (lons_rad[j] - lons_rad[i]) * (2 + np.sin(lats_rad[i]) + np.sin(lats_rad[j]))
    
    area = abs(area) / 2.0
    
    # Convert to km²
    R = 6371.0  # Earth radius in km
    area_km2 = area * R * R
    
    return area_km2

def point_in_polygon(lat: float, lon: float, 
                    polygon_lats: np.ndarray, polygon_lons: np.ndarray) -> bool:
    """
    Test if a point is inside a polygon using ray casting algorithm.
    
    Args:
        lat, lon: Point coordinates (degrees)
        polygon_lats, polygon_lons: Polygon vertex coordinates (degrees)
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    
    n = len(polygon_lats)
    inside = False
    
    p1_lat, p1_lon = polygon_lats[0], polygon_lons[0]
    
    for i in range(1, n + 1):
        p2_lat, p2_lon = polygon_lats[i % n], polygon_lons[i % n]
        
        if lat > min(p1_lat, p2_lat):
            if lat <= max(p1_lat, p2_lat):
                if lon <= max(p1_lon, p2_lon):
                    if p1_lat != p2_lat:
                        lon_intersect = (lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                    if p1_lon == p2_lon or lon <= lon_intersect:
                        inside = not inside
        
        p1_lat, p1_lon = p2_lat, p2_lon
    
    return inside

def create_buffer_around_point(lat: float, lon: float, 
                              radius_km: float, n_points: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a circular buffer around a point.
    
    Args:
        lat, lon: Center point coordinates (degrees)
        radius_km: Buffer radius in kilometers
        n_points: Number of points to define the circle
        
    Returns:
        Tuple of (buffer_lats, buffer_lons) arrays
    """
    
    # Earth radius in km
    R = 6371.0
    
    # Convert center to radians
    lat_rad = radians(lat)
    lon_rad = radians(lon)
    
    # Angular distance
    angular_distance = radius_km / R
    
    # Generate points around the circle
    bearings = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    buffer_lats = []
    buffer_lons = []
    
    for bearing in bearings:
        # Calculate point at given bearing and distance
        lat2_rad = asin(sin(lat_rad) * cos(angular_distance) +
                       cos(lat_rad) * sin(angular_distance) * cos(bearing))
        
        lon2_rad = lon_rad + atan2(sin(bearing) * sin(angular_distance) * cos(lat_rad),
                                  cos(angular_distance) - sin(lat_rad) * sin(lat2_rad))
        
        buffer_lats.append(degrees(lat2_rad))
        buffer_lons.append(degrees(lon2_rad))
    
    return np.array(buffer_lats), np.array(buffer_lons)

def calculate_grid_cell_area(lat: float, lat_spacing: float, lon_spacing: float) -> float:
    """
    Calculate the area of a grid cell at given latitude.
    
    Args:
        lat: Latitude of grid cell center (degrees)
        lat_spacing: Grid spacing in latitude direction (degrees)
        lon_spacing: Grid spacing in longitude direction (degrees)
        
    Returns:
        Grid cell area in km²
    """
    
    # Earth radius in km
    R = 6371.0
    
    # Convert to radians
    lat_rad = radians(lat)
    lat_spacing_rad = radians(lat_spacing)
    lon_spacing_rad = radians(lon_spacing)
    
    # Calculate area using spherical approximation
    area = R * R * abs(lon_spacing_rad) * abs(sin(lat_rad + lat_spacing_rad/2) - 
                                             sin(lat_rad - lat_spacing_rad/2))
    
    return area

def find_nearest_grid_point(target_lat: float, target_lon: float,
                           grid_lats: np.ndarray, grid_lons: np.ndarray) -> Tuple[int, int]:
    """
    Find the nearest grid point to target coordinates.
    
    Args:
        target_lat, target_lon: Target coordinates (degrees)
        grid_lats, grid_lons: Grid coordinate arrays
        
    Returns:
        Tuple of (lat_index, lon_index) of nearest grid point
    """
    
    # Calculate distances to all grid points
    distances = np.sqrt((grid_lats - target_lat)**2 + (grid_lons - target_lon)**2)
    
    # Find minimum distance
    lat_idx, lon_idx = np.unravel_index(distances.argmin(), distances.shape)
    
    return lat_idx, lon_idx

def interpolate_to_grid(source_lats: np.ndarray, source_lons: np.ndarray, source_values: np.ndarray,
                       target_lats: np.ndarray, target_lons: np.ndarray,
                       method: str = 'linear') -> np.ndarray:
    """
    Interpolate scattered data to regular grid.
    
    Args:
        source_lats, source_lons: Source coordinate arrays
        source_values: Values at source coordinates
        target_lats, target_lons: Target grid coordinates
        method: Interpolation method ('linear', 'cubic', 'nearest')
        
    Returns:
        Interpolated values on target grid
    """
    
    from scipy.interpolate import griddata
    
    # Remove NaN values
    valid_mask = ~np.isnan(source_values)
    
    if np.sum(valid_mask) < 3:
        logger.warning("Insufficient valid data for interpolation")
        return np.full(target_lats.shape, np.nan)
    
    source_points = np.column_stack((source_lats[valid_mask], source_lons[valid_mask]))
    target_points = np.column_stack((target_lats.ravel(), target_lons.ravel()))
    
    try:
        interpolated = griddata(source_points, source_values[valid_mask], 
                              target_points, method=method, fill_value=np.nan)
        return interpolated.reshape(target_lats.shape)
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        return np.full(target_lats.shape, np.nan)

def calculate_spatial_gradient(data: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spatial gradients of a 2D field.
    
    Args:
        data: 2D data array
        lats, lons: Coordinate arrays
        
    Returns:
        Tuple of (gradient_lat, gradient_lon) arrays
    """
    
    # Calculate grid spacing
    if len(lats) > 1:
        dlat = np.mean(np.diff(lats))
    else:
        dlat = 1.0
        
    if len(lons) > 1:
        dlon = np.mean(np.diff(lons))
    else:
        dlon = 1.0
    
    # Convert to meters for gradient calculation
    lat_center = np.mean(lats)
    dlat_m = dlat * 111320.0  # meters per degree latitude
    dlon_m = dlon * 111320.0 * cos(radians(lat_center))  # meters per degree longitude
    
    # Calculate gradients
    grad_lat, grad_lon = np.gradient(data, dlat_m, dlon_m)
    
    return grad_lat, grad_lon

def spatial_distance_matrix(lats1: np.ndarray, lons1: np.ndarray,
                           lats2: Optional[np.ndarray] = None, 
                           lons2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate distance matrix between two sets of coordinates.
    
    Args:
        lats1, lons1: First set of coordinates
        lats2, lons2: Second set of coordinates (optional, defaults to first set)
        
    Returns:
        Distance matrix in kilometers
    """
    
    if lats2 is None:
        lats2 = lats1
        lons2 = lons1
    
    n1 = len(lats1)
    n2 = len(lats2)
    
    distances = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            distances[i, j] = haversine_distance(lats1[i], lons1[i], lats2[j], lons2[j])
    
    return distances

def create_convex_hull(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create convex hull of a set of points.
    
    Args:
        lats, lons: Point coordinates
        
    Returns:
        Tuple of (hull_lats, hull_lons) defining convex hull
    """
    
    try:
        from scipy.spatial import ConvexHull
        
        points = np.column_stack((lats, lons))
        hull = ConvexHull(points)
        
        hull_points = points[hull.vertices]
        return hull_points[:, 0], hull_points[:, 1]
        
    except ImportError:
        logger.warning("SciPy not available for convex hull calculation")
        # Return bounding box as fallback
        return np.array([lats.min(), lats.min(), lats.max(), lats.max()]), \
               np.array([lons.min(), lons.max(), lons.max(), lons.min()])

def calculate_centroid(lats: np.ndarray, lons: np.ndarray, 
                      weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Calculate centroid of a set of points.
    
    Args:
        lats, lons: Point coordinates
        weights: Optional weights for each point
        
    Returns:
        Tuple of (centroid_lat, centroid_lon)
    """
    
    if weights is None:
        weights = np.ones(len(lats))
    
    # Remove NaN values
    valid_mask = ~(np.isnan(lats) | np.isnan(lons) | np.isnan(weights))
    
    if np.sum(valid_mask) == 0:
        return np.nan, np.nan
    
    lats_valid = lats[valid_mask]
    lons_valid = lons[valid_mask]
    weights_valid = weights[valid_mask]
    
    # Weighted centroid
    centroid_lat = np.average(lats_valid, weights=weights_valid)
    centroid_lon = np.average(lons_valid, weights=weights_valid)
    
    return centroid_lat, centroid_lon

def transform_coordinates(lats: np.ndarray, lons: np.ndarray,
                         source_crs: str = 'EPSG:4326',
                         target_crs: str = 'EPSG:3857') -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform coordinates between coordinate reference systems.
    
    Args:
        lats, lons: Input coordinates
        source_crs: Source CRS (default: WGS84)
        target_crs: Target CRS (default: Web Mercator)
        
    Returns:
        Tuple of transformed (x, y) coordinates
    """
    
    try:
        import pyproj
        
        # Create transformer
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
        
        # Transform coordinates
        x, y = transformer.transform(lons, lats)
        
        return x, y
        
    except ImportError:
        logger.warning("pyproj not available, using simple projection")
        # Fallback to simple mercator projection
        return project_coordinates(lats, lons, 'mercator')

def validate_coordinates(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Validate coordinate arrays and return validity mask.
    
    Args:
        lats, lons: Coordinate arrays to validate
        
    Returns:
        Tuple of (validity_mask, all_valid_flag)
    """
    
    # Check for valid latitude range
    lat_valid = (lats >= -90) & (lats <= 90)
    
    # Check for valid longitude range
    lon_valid = (lons >= -180) & (lons <= 180)
    
    # Check for NaN values
    not_nan = ~(np.isnan(lats) | np.isnan(lons))
    
    # Combined validity
    valid_mask = lat_valid & lon_valid & not_nan
    all_valid = np.all(valid_mask)
    
    if not all_valid:
        n_invalid = np.sum(~valid_mask)
        logger.warning(f"Found {n_invalid} invalid coordinates")
    
    return valid_mask, all_valid

def create_spatial_index(lats: np.ndarray, lons: np.ndarray, data: Optional[np.ndarray] = None) -> Dict:
    """
    Create a spatial index for efficient spatial queries.
    
    Args:
        lats, lons: Coordinate arrays
        data: Optional data associated with each coordinate
        
    Returns:
        Dictionary containing spatial index information
    """
    
    # Simple grid-based spatial index
    # For more sophisticated indexing, could use rtree or similar
    
    lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
    lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
    
    # Create grid cells (adjust resolution as needed)
    n_cells = int(np.sqrt(len(lats))) + 1
    lat_edges = np.linspace(lat_min, lat_max, n_cells + 1)
    lon_edges = np.linspace(lon_min, lon_max, n_cells + 1)
    
    # Assign points to grid cells
    lat_indices = np.digitize(lats, lat_edges) - 1
    lon_indices = np.digitize(lons, lon_edges) - 1
    
    # Create index dictionary
    spatial_index = {
        'grid_shape': (n_cells, n_cells),
        'lat_edges': lat_edges,
        'lon_edges': lon_edges,
        'point_indices': {},
        'bounds': (lat_min, lat_max, lon_min, lon_max)
    }
    
    # Populate index
    for i, (lat_idx, lon_idx) in enumerate(zip(lat_indices, lon_indices)):
        # Ensure indices are within bounds
        lat_idx = np.clip(lat_idx, 0, n_cells - 1)
        lon_idx = np.clip(lon_idx, 0, n_cells - 1)
        
        cell_key = (lat_idx, lon_idx)
        if cell_key not in spatial_index['point_indices']:
            spatial_index['point_indices'][cell_key] = []
        
        point_info = {'index': i, 'lat': lats[i], 'lon': lons[i]}
        if data is not None:
            point_info['data'] = data[i]
        
        spatial_index['point_indices'][cell_key].append(point_info)
    
    return spatial_index

def query_spatial_index(spatial_index: Dict, query_lat: float, query_lon: float,
                       radius_km: float) -> List[Dict]:
    """
    Query spatial index for points within radius of query point.
    
    Args:
        spatial_index: Spatial index created by create_spatial_index
        query_lat, query_lon: Query point coordinates
        radius_km: Search radius in kilometers
        
    Returns:
        List of points within radius
    """
    
    lat_edges = spatial_index['lat_edges']
    lon_edges = spatial_index['lon_edges']
    
    # Find relevant grid cells
    # Convert radius to approximate degree range
    lat_range = radius_km / 111.32
    lon_range = radius_km / (111.32 * cos(radians(query_lat)))
    
    lat_min_search = query_lat - lat_range
    lat_max_search = query_lat + lat_range
    lon_min_search = query_lon - lon_range
    lon_max_search = query_lon + lon_range
    
    # Find grid cell indices to search
    lat_idx_min = max(0, np.searchsorted(lat_edges, lat_min_search) - 1)
    lat_idx_max = min(len(lat_edges) - 1, np.searchsorted(lat_edges, lat_max_search))
    lon_idx_min = max(0, np.searchsorted(lon_edges, lon_min_search) - 1)
    lon_idx_max = min(len(lon_edges) - 1, np.searchsorted(lon_edges, lon_max_search))
    
    # Search relevant cells
    nearby_points = []
    
    for lat_idx in range(lat_idx_min, lat_idx_max + 1):
        for lon_idx in range(lon_idx_min, lon_idx_max + 1):
            cell_key = (lat_idx, lon_idx)
            
            if cell_key in spatial_index['point_indices']:
                for point in spatial_index['point_indices'][cell_key]:
                    distance = haversine_distance(query_lat, query_lon, 
                                                point['lat'], point['lon'])
                    
                    if distance <= radius_km:
                        point_copy = point.copy()
                        point_copy['distance_km'] = distance
                        nearby_points.append(point_copy)
    
    # Sort by distance
    nearby_points.sort(key=lambda x: x['distance_km'])
    
    return nearby_points