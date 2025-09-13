"""
Corelia Training Package

AI-powered medical note processing for French healthcare.
"""

__version__ = "0.1.0"
__author__ = "Hicham"
__email__ = "dhicham.pro@gmail.com"

from .services import DatasetLoader, ModelTrainer, ModelEvaluator, ModelSaver
from .utils import setup_logging, log_to_mlflow, load_training_config

__all__ = [
    "DatasetLoader",
    "ModelTrainer", 
    "ModelEvaluator",
    "ModelSaver",
    "setup_logging",
    "log_to_mlflow",
    "load_training_config"
]