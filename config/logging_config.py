# ============================================================================
# FILE: config/logging_config.py
# ============================================================================
import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_dir: str = "logs",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 console_output: bool = True,
                 format_style: str = "detailed") -> logging.Logger:
    """
    Setup comprehensive logging configuration for the super-emitter tracking system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file path (optional)
        log_dir: Directory for log files
        max_file_size: Maximum size of log file before rotation (bytes)
        backup_count: Number of backup files to keep
        console_output: Whether to output logs to console
        format_style: Log format style ('simple', 'detailed', 'json')
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set up log file path
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir_path / f"super_emitter_tracking_{timestamp}.log"
    else:
        log_file = Path(log_file)
        # Ensure the directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    formatters = _create_formatters(format_style)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatters['detailed'])
    root_logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatters['console'])
        root_logger.addHandler(console_handler)
    
    # Add error file handler for ERROR and CRITICAL messages
    error_log_file = log_dir_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        filename=error_log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatters['detailed'])
    root_logger.addHandler(error_handler)
    
    # Configure specific loggers for different modules
    _configure_module_loggers(numeric_level)
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("LOGGING CONFIGURATION INITIALIZED")
    logger.info("=" * 60)
    logger.info(f"Log level: {log_level}")
    logger.info(f"Main log file: {log_file}")
    logger.info(f"Error log file: {error_log_file}")
    logger.info(f"Console output: {console_output}")
    logger.info(f"Max file size: {max_file_size / (1024*1024):.1f} MB")
    logger.info(f"Backup count: {backup_count}")
    
    return logger

def _create_formatters(format_style: str) -> Dict[str, logging.Formatter]:
    """Create different log formatters based on style."""
    
    formatters = {}
    
    if format_style == "simple":
        # Simple format for development
        simple_format = "%(levelname)s - %(name)s - %(message)s"
        formatters['console'] = logging.Formatter(simple_format)
        formatters['detailed'] = logging.Formatter(simple_format)
        
    elif format_style == "json":
        # JSON format for structured logging
        json_format = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
        formatters['console'] = logging.Formatter(json_format)
        formatters['detailed'] = logging.Formatter(json_format)
        
    else:  # detailed (default)
        # Detailed format for production
        detailed_format = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
        console_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        
        formatters['detailed'] = logging.Formatter(
            detailed_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        formatters['console'] = logging.Formatter(
            console_format,
            datefmt="%H:%M:%S"
        )
    
    return formatters

def _configure_module_loggers(level: int):
    """Configure specific loggers for different modules."""
    
    # Super-emitter tracking modules
    module_configs = {
        'src.data': level,
        'src.detection': level,
        'src.tracking': level,
        'src.analysis': level,
        'src.visualization': level,
        'src.alerts': level,
        'src.utils': level,
    }
    
    # External library loggers (usually more verbose)
    external_configs = {
        'earthengine': logging.WARNING,  # Earth Engine can be very verbose
        'urllib3': logging.WARNING,      # HTTP requests
        'requests': logging.WARNING,     # HTTP requests
        'matplotlib': logging.WARNING,   # Plotting library
        'folium': logging.WARNING,       # Map library
        'streamlit': logging.WARNING,    # Dashboard
        'googleapiclient': logging.WARNING,  # Google API client
    }
    
    # Apply module configurations
    for module_name, module_level in module_configs.items():
        logger = logging.getLogger(module_name)
        logger.setLevel(module_level)
    
    # Apply external library configurations
    for module_name, module_level in external_configs.items():
        logger = logging.getLogger(module_name)
        logger.setLevel(module_level)

def create_performance_logger() -> logging.Logger:
    """Create a separate logger for performance monitoring."""
    
    perf_logger = logging.getLogger('performance')
    
    # Create performance log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    perf_log_file = log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Performance-specific formatter
    perf_formatter = logging.Formatter(
        "%(asctime)s - PERF - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler for performance logs
    perf_handler = logging.handlers.RotatingFileHandler(
        filename=perf_log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    perf_handler.setFormatter(perf_formatter)
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    # Prevent propagation to root logger
    perf_logger.propagate = False
    
    return perf_logger

def create_audit_logger() -> logging.Logger:
    """Create a separate logger for audit trail (user actions, data changes)."""
    
    audit_logger = logging.getLogger('audit')
    
    # Create audit log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    audit_log_file = log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Audit-specific formatter (more structured)
    audit_formatter = logging.Formatter(
        "%(asctime)s - AUDIT - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler for audit logs
    audit_handler = logging.handlers.RotatingFileHandler(
        filename=audit_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10  # Keep more audit logs
    )
    audit_handler.setFormatter(audit_formatter)
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    
    # Prevent propagation to root logger
    audit_logger.propagate = False
    
    return audit_logger

def log_function_call(func):
    """Decorator to log function calls with execution time."""
    
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        # Log function entry
        logger.debug(f"Entering {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful completion
            logger.debug(f"Completed {func.__name__} in {execution_time:.3f} seconds")
            
            # Log to performance logger if slow
            if execution_time > 1.0:  # Log functions taking > 1 second
                perf_logger = logging.getLogger('performance')
                perf_logger.info(f"{func.__module__}.{func.__name__} took {execution_time:.3f} seconds")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.3f} seconds: {str(e)}")
            raise
    
    return wrapper

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)

def log_system_info():
    """Log system information for debugging purposes."""
    
    import platform
    import psutil
    
    logger = logging.getLogger(__name__)
    
    logger.info("SYSTEM INFORMATION:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python version: {platform.python_version()}")
    logger.info(f"  CPU cores: {psutil.cpu_count()}")
    logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"  Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")

def setup_development_logging():
    """Quick setup for development with debug level and console output."""
    return setup_logging(
        log_level="DEBUG",
        console_output=True,
        format_style="simple"
    )

def setup_production_logging(log_dir: str = "/var/log/super-emitter-tracking"):
    """Setup for production with appropriate log rotation and error handling."""
    return setup_logging(
        log_level="INFO",
        log_dir=log_dir,
        console_output=False,
        format_style="detailed",
        max_file_size=50 * 1024 * 1024,  # 50MB
        backup_count=10
    )