"""Logging configuration for the TwCS Topic Modeling system."""
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Global unified debug log file
_UNIFIED_DEBUG_LOG = None
_UNIFIED_HANDLER = None


def get_unified_debug_log_path() -> str:
    """
    Get the path to the unified debug log file.
    
    Uses UNIFIED_DEBUG_LOG environment variable if set (from run_full_system.sh),
    otherwise creates a new log file in logs/debug/ directory.
    """
    global _UNIFIED_DEBUG_LOG
    if _UNIFIED_DEBUG_LOG is None:
        # Check if shell script already set the log path
        env_log_path = os.environ.get('UNIFIED_DEBUG_LOG')
        if env_log_path:
            _UNIFIED_DEBUG_LOG = env_log_path
        else:
            # Create new log in debug folder
            log_dir = Path("logs/debug")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            _UNIFIED_DEBUG_LOG = str(log_dir / f"run_{timestamp}.log")
    return _UNIFIED_DEBUG_LOG


def get_unified_handler() -> logging.FileHandler:
    """Get or create the unified debug file handler."""
    global _UNIFIED_HANDLER
    if _UNIFIED_HANDLER is None:
        log_path = get_unified_debug_log_path()
        _UNIFIED_HANDLER = logging.FileHandler(log_path)
        _UNIFIED_HANDLER.setLevel(logging.DEBUG)
        
        # Detailed format showing component, function, and line
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s | L%(lineno)-4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        _UNIFIED_HANDLER.setFormatter(formatter)
        
        # Write header to the log file
        with open(log_path, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write(f"UNIFIED DEBUG LOG - Started at {datetime.now().isoformat()}\n")
            f.write("=" * 120 + "\n")
            f.write("Format: TIMESTAMP | LEVEL | COMPONENT | FUNCTION | LINE | MESSAGE\n")
            f.write("=" * 120 + "\n\n")
    
    return _UNIFIED_HANDLER


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console, file, and unified debug handlers.
    
    Args:
        name: Logger name
        log_file: Log file path (optional, for component-specific logs)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Component-specific file handler (if log_file provided)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Add unified debug handler (all logs go here)
    unified_handler = get_unified_handler()
    logger.addHandler(unified_handler)
    
    # Log that this component has been initialized
    logger.debug(f"Logger initialized for component: {name}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger by name."""
    return logging.getLogger(name)


def log_flow_marker(marker: str, component: str = "SYSTEM"):
    """
    Log a flow marker to the unified debug log.
    Use this to mark major execution phases.
    
    Args:
        marker: Description of the flow phase
        component: Component name
    """
    logger = logging.getLogger(component)
    if not logger.handlers:
        logger = setup_logger(component)
    
    separator = "=" * 80
    logger.info(separator)
    logger.info(f">>> FLOW: {marker}")
    logger.info(separator)


def log_step(step_num: int, description: str, component: str = "SYSTEM"):
    """
    Log a numbered step in the execution flow.
    
    Args:
        step_num: Step number
        description: Step description
        component: Component name
    """
    logger = logging.getLogger(component)
    if not logger.handlers:
        logger = setup_logger(component)
    
    logger.info(f"[STEP {step_num}] {description}")


def log_debug_checkpoint(checkpoint: str, data: dict = None, component: str = "DEBUG"):
    """
    Log a debug checkpoint with optional data.
    
    Args:
        checkpoint: Checkpoint description
        data: Optional data dictionary to log
        component: Component name
    """
    logger = logging.getLogger(component)
    if not logger.handlers:
        logger = setup_logger(component)
    
    logger.debug(f"🔍 CHECKPOINT: {checkpoint}")
    if data:
        for key, value in data.items():
            if hasattr(value, '__len__') and not isinstance(value, str):
                logger.debug(f"    {key}: {type(value).__name__} with {len(value)} items")
            else:
                logger.debug(f"    {key}: {value}")

