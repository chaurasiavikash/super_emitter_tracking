# ============================================================================
# FILE: src/utils/file_utils.py
# ============================================================================
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point

logger = logging.getLogger(__name__)

class FileManager:
    """
    Comprehensive file management utilities for the super-emitter tracking system.
    
    Handles:
    - Directory creation and management
    - Data export in multiple formats
    - Configuration file handling
    - Log file management
    - Backup and archiving
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_output_path = Path(config.get('processing', {}).get('output', {}).get('base_path', './data/outputs'))
        
    def create_output_directories(self, base_path: Union[str, Path]) -> None:
        """Create all necessary output directories."""
        
        base_path = Path(base_path)
        
        directories = [
            'detections',
            'tracking',
            'analysis',
            'alerts',
            'data',
            'visualizations',
            'reports',
            'exports'
        ]
        
        for directory in directories:
            dir_path = base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
        
        logger.info(f"Output directories created at: {base_path}")
    
    def save_dataframe(self, df: pd.DataFrame, file_path: Union[str, Path],
                      format_type: str = 'csv', **kwargs) -> bool:
        """Save DataFrame to file in specified format."""
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format_type.lower() == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif format_type.lower() == 'json':
                df.to_json(file_path, orient='records', indent=2, **kwargs)
            elif format_type.lower() == 'parquet':
                df.to_parquet(file_path, **kwargs)
            elif format_type.lower() == 'excel':
                df.to_excel(file_path, index=False, **kwargs)
            elif format_type.lower() == 'hdf5':
                df.to_hdf(file_path, key='data', mode='w', **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"DataFrame saved to {file_path} ({format_type.upper()})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save DataFrame to {file_path}: {e}")
            return False
    
    def save_geojson(self, df: pd.DataFrame, file_path: Union[str, Path],
                    lat_col: str = 'lat', lon_col: str = 'lon') -> bool:
        """Save DataFrame as GeoJSON with geometry from lat/lon columns."""
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create geometry from lat/lon
            geometry = [Point(lon, lat) for lat, lon in zip(df[lat_col], df[lon_col])]
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df.drop(columns=[lat_col, lon_col]), 
                                  geometry=geometry, crs='EPSG:4326')
            
            # Save as GeoJSON
            gdf.to_file(file_path, driver='GeoJSON')
            
            logger.info(f"GeoJSON saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save GeoJSON to {file_path}: {e}")
            return False
    
    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """Save dictionary as JSON file."""
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=self._json_serializer)
            
            logger.info(f"JSON saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}")
            return False
    
    def load_json(self, file_path: Union[str, Path]) -> Optional[Dict]:
        """Load JSON file as dictionary."""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"JSON file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"JSON loaded from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSON from {file_path}: {e}")
            return None
    
    def save_netcdf(self, dataset: xr.Dataset, file_path: Union[str, Path]) -> bool:
        """Save xarray Dataset as NetCDF file."""
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Add metadata
            dataset.attrs['created'] = pd.Timestamp.now().isoformat()
            dataset.attrs['source'] = 'super-emitter-tracking-system'
            
            # Save with compression
            encoding = {}
            for var in dataset.data_vars:
                if dataset[var].dtype == 'float64':
                    encoding[var] = {'dtype': 'float32', 'zlib': True, 'complevel': 6}
                elif dataset[var].dtype == 'int64':
                    encoding[var] = {'dtype': 'int32', 'zlib': True, 'complevel': 6}
            
            dataset.to_netcdf(file_path, encoding=encoding)
            
            logger.info(f"NetCDF saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save NetCDF to {file_path}: {e}")
            return False
    
    def load_dataframe(self, file_path: Union[str, Path], 
                      format_type: str = 'auto') -> Optional[pd.DataFrame]:
        """Load DataFrame from file, auto-detecting format if needed."""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        # Auto-detect format from extension
        if format_type == 'auto':
            suffix = file_path.suffix.lower()
            format_map = {
                '.csv': 'csv',
                '.json': 'json',
                '.parquet': 'parquet',
                '.xlsx': 'excel',
                '.xls': 'excel',
                '.h5': 'hdf5',
                '.hdf5': 'hdf5'
            }
            format_type = format_map.get(suffix, 'csv')
        
        try:
            if format_type == 'csv':
                df = pd.read_csv(file_path)
            elif format_type == 'json':
                df = pd.read_json(file_path)
            elif format_type == 'parquet':
                df = pd.read_parquet(file_path)
            elif format_type == 'excel':
                df = pd.read_excel(file_path)
            elif format_type == 'hdf5':
                df = pd.read_hdf(file_path, key='data')
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"DataFrame loaded from {file_path} ({format_type.upper()})")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load DataFrame from {file_path}: {e}")
            return None
    
    def backup_file(self, file_path: Union[str, Path], 
                   backup_dir: Optional[Union[str, Path]] = None) -> bool:
        """Create backup copy of a file."""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File to backup not found: {file_path}")
            return False
        
        # Default backup directory
        if backup_dir is None:
            backup_dir = file_path.parent / 'backups'
        
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped backup filename
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            
            logger.info(f"Backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def compress_directory(self, dir_path: Union[str, Path], 
                          output_path: Optional[Union[str, Path]] = None) -> bool:
        """Compress directory to ZIP archive."""
        
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return False
        
        if output_path is None:
            output_path = dir_path.with_suffix('.zip')
        
        output_path = Path(output_path)
        
        try:
            import shutil
            shutil.make_archive(str(output_path.with_suffix('')), 'zip', str(dir_path))
            
            logger.info(f"Directory compressed to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to compress directory: {e}")
            return False
    
    def cleanup_old_files(self, directory: Union[str, Path], 
                         max_age_days: int = 30, pattern: str = "*") -> int:
        """Clean up old files in directory."""
        
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return 0
        
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
        deleted_count = 0
        
        try:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_time = pd.Timestamp.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_time < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old file: {file_path}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old files from {directory}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def get_directory_size(self, directory: Union[str, Path]) -> float:
        """Get total size of directory in MB."""
        
        directory = Path(directory)
        
        if not directory.exists():
            return 0.0
        
        total_size = 0
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Error calculating directory size: {e}")
            return 0.0
    
    def export_multiple_formats(self, df: pd.DataFrame, base_path: Union[str, Path],
                               formats: list = None) -> Dict[str, bool]:
        """Export DataFrame to multiple formats."""
        
        if formats is None:
            formats = ['csv', 'json', 'parquet']
        
        base_path = Path(base_path)
        results = {}
        
        for fmt in formats:
            if fmt == 'geojson' and 'center_lat' in df.columns and 'center_lon' in df.columns:
                success = self.save_geojson(df, base_path.with_suffix('.geojson'))
            else:
                file_path = base_path.with_suffix(f'.{fmt}')
                success = self.save_dataframe(df, file_path, fmt)
            
            results[fmt] = success
        
        return results
    
    def create_archive(self, files: list, archive_path: Union[str, Path]) -> bool:
        """Create ZIP archive from list of files."""
        
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import zipfile
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in files:
                    file_path = Path(file_path)
                    if file_path.exists():
                        # Use relative path in archive
                        arcname = file_path.name
                        zf.write(file_path, arcname)
                        logger.debug(f"Added to archive: {file_path}")
            
            logger.info(f"Archive created: {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create archive: {e}")
            return False
    
    def validate_file_integrity(self, file_path: Union[str, Path]) -> bool:
        """Validate file integrity by attempting to read it."""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False
        
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.csv':
                pd.read_csv(file_path, nrows=1)
            elif suffix == '.json':
                with open(file_path, 'r') as f:
                    json.load(f)
            elif suffix == '.parquet':
                pd.read_parquet(file_path, max_rows_per_file=1)
            elif suffix in ['.xlsx', '.xls']:
                pd.read_excel(file_path, nrows=1)
            elif suffix == '.nc':
                xr.open_dataset(file_path)
            elif suffix in ['.h5', '.hdf5']:
                pd.read_hdf(file_path, key='data', stop=1)
            
            return True
            
        except Exception as e:
            logger.warning(f"File integrity check failed for {file_path}: {e}")
            return False
    
    def get_file_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive metadata for a file."""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {}
        
        stat = file_path.stat()
        
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_extension': file_path.suffix,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created_time': pd.Timestamp.fromtimestamp(stat.st_ctime),
            'modified_time': pd.Timestamp.fromtimestamp(stat.st_mtime),
            'is_valid': self.validate_file_integrity(file_path)
        }
        
        # Try to get additional format-specific metadata
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.csv':
                df = pd.read_csv(file_path, nrows=0)  # Just headers
                metadata.update({
                    'columns': list(df.columns),
                    'column_count': len(df.columns),
                    'estimated_rows': self._estimate_csv_rows(file_path)
                })
            
            elif suffix == '.nc':
                ds = xr.open_dataset(file_path)
                metadata.update({
                    'data_vars': list(ds.data_vars.keys()),
                    'coordinates': list(ds.coords.keys()),
                    'dimensions': dict(ds.dims),
                    'attributes': dict(ds.attrs)
                })
                ds.close()
            
        except Exception as e:
            logger.debug(f"Could not get extended metadata for {file_path}: {e}")
        
        return metadata
    
    def _estimate_csv_rows(self, file_path: Path) -> int:
        """Estimate number of rows in CSV file."""
        try:
            with open(file_path, 'r') as f:
                # Count newlines in first chunk
                chunk = f.read(8192)
                if not chunk:
                    return 0
                
                lines_in_chunk = chunk.count('\n')
                
                # Estimate total lines based on file size
                file_size = file_path.stat().st_size
                estimated_lines = int((lines_in_chunk / len(chunk)) * file_size)
                
                # Subtract 1 for header
                return max(0, estimated_lines - 1)
                
        except Exception:
            return 0
    
    def _json_serializer(self, obj):
        """JSON serializer for non-standard types."""
        
        if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def create_data_catalog(self, directory: Union[str, Path]) -> pd.DataFrame:
        """Create catalog of all data files in directory."""
        
        directory = Path(directory)
        
        if not directory.exists():
            return pd.DataFrame()
        
        catalog_data = []
        
        # Scan all files
        for file_path in directory.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                metadata = self.get_file_metadata(file_path)
                
                # Add relative path
                metadata['relative_path'] = str(file_path.relative_to(directory))
                
                catalog_data.append(metadata)
        
        if catalog_data:
            catalog_df = pd.DataFrame(catalog_data)
            
            # Save catalog
            catalog_path = directory / 'data_catalog.csv'
            self.save_dataframe(catalog_df, catalog_path)
            
            logger.info(f"Data catalog created with {len(catalog_df)} files")
            return catalog_df
        
        return pd.DataFrame()
    
    def sync_to_remote(self, local_path: Union[str, Path], 
                      remote_config: Dict) -> bool:
        """Sync local directory to remote storage (placeholder for cloud sync)."""
        
        # This would implement actual cloud sync (S3, GCS, etc.)
        # For now, just log the operation
        
        logger.info(f"Sync requested: {local_path} -> {remote_config.get('destination', 'remote')}")
        
        # In real implementation, would use boto3, gsutil, etc.
        # return upload_to_cloud(local_path, remote_config)
        
        return True  # Mock success