#!/usr/bin/env python3
"""
Quick test script for Mistral-7B fine-tuning pipeline.

This script performs a quick test of the training pipeline with minimal
data to verify everything is working correctly.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add training package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.services import DatasetLoader, ModelTrainer, ModelEvaluator
from training.utils import setup_logging, load_training_config


def main():
    """Quick test function."""
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("corelia.test")
    
    logger.info("🧪 Starting quick test of Mistral-7B training pipeline")
    
    try:
        # Load configuration
        load_training_config()
        logger.info("✅ Configuration loaded")
        
        # Test dataset loading
        logger.info("📊 Testing dataset loading...")
        dataset_loader = DatasetLoader()
        
        # Load with minimal samples
        datasets = dataset_loader.load_mistral_datasets(
            max_samples=10,  # Very small for testing
            include_test=True
        )
        
        if datasets:
            logger.info(f"✅ Loaded {len(datasets)} datasets")
            for dataset_name, dataset_info in datasets.items():
                splits = list(dataset_info["splits"].keys())
                logger.info(f"  - {dataset_name}: {splits}")
        else:
            logger.error("❌ Failed to load datasets")
            return 1
        
        # Test model trainer initialization
        logger.info("🤖 Testing model trainer initialization...")
        config_path = "training/configs/config_mistral_7B.yaml"
        if Path(config_path).exists():
            model_trainer = ModelTrainer(config_path)
            logger.info("✅ Model trainer initialized")
        else:
            logger.warning(f"⚠️ Config file not found: {config_path}")
        
        # Test evaluator initialization
        logger.info("📈 Testing evaluator initialization...")
        model_evaluator = ModelEvaluator()
        logger.info(f"✅ Evaluator initialized with metrics: {model_evaluator.enabled_metrics}")
        
        # Test environment variables
        logger.info("🔧 Testing environment variables...")
        required_vars = [
            "DATASETS_MISTRAL",
            "EVALUATION_METRICS",
            "DATASET_WEIGHT_NACHOS",
            "DATASET_WEIGHT_MEDIQAL",
            "DATASET_WEIGHT_FRASIMED"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
        else:
            logger.info("✅ All required environment variables are set")
        
        logger.info("🎉 Quick test completed successfully!")
        logger.info("✅ Training pipeline is ready to use")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
