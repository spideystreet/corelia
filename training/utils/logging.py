"""
Logging utilities for training.
"""

import logging
import mlflow
from typing import Dict, Any


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger("corelia.training")
    return logger


def log_to_mlflow(metrics: Dict[str, Any], step: int = None) -> None:
    """
    Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Step number for the metrics
    """
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_text(str(value), f"{key}.txt")
