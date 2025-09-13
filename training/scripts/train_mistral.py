#!/usr/bin/env python3
"""
Main training script for Mistral-7B fine-tuning with LoRA.

This script orchestrates the complete training pipeline:
1. Load datasets from Hugging Face Hub
2. Setup model and LoRA configuration
3. Train the model with MLflow tracking
4. Evaluate the model
5. Save to MLflow Model Registry
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add training package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.services import DatasetLoader, ModelTrainer, ModelEvaluator, ModelSaver
from training.utils import setup_logging, load_training_config


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train Mistral-7B for French medical text processing"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/config_mistral_7B.yaml",
        help="Path to training configuration YAML file"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset for testing (default: None for full dataset)"
    )
    
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include test datasets for evaluation"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="corelia-mistral-7b-french-medical",
        help="Name for the trained model"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for saved models"
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip model evaluation after training"
    )
    
    parser.add_argument(
        "--skip-mlflow-save",
        action="store_true",
        help="Skip saving to MLflow Model Registry"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def main():
    """Main training function."""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger("corelia.training.main")
    
    logger.info("🚀 Starting Mistral-7B fine-tuning pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Model name: {args.model_name}")
    
    try:
        # Load training configuration
        load_training_config()
        logger.info("✅ Training configuration loaded")
        
        # Initialize services
        logger.info("📊 Initializing services...")
        dataset_loader = DatasetLoader()
        model_trainer = ModelTrainer(args.config)
        model_evaluator = ModelEvaluator()
        model_saver = ModelSaver()
        
        # Load datasets
        logger.info("📚 Loading datasets...")
        datasets = dataset_loader.load_mistral_datasets(
            max_samples=args.max_samples,
            include_test=args.include_test
        )
        
        if not datasets:
            logger.error("❌ No datasets loaded successfully")
            return 1
        
        logger.info(f"✅ Loaded {len(datasets)} datasets")
        for dataset_name, dataset_info in datasets.items():
            splits = list(dataset_info["splits"].keys())
            logger.info(f"  - {dataset_name}: {splits} (weight: {dataset_info['weight']})")
        
        # Prepare training data
        logger.info("🔄 Preparing training data...")
        train_datasets = []
        eval_datasets = []
        
        for dataset_name, dataset_info in datasets.items():
            splits = dataset_info["splits"]
            
            if "train" in splits:
                train_datasets.append(splits["train"])
                logger.info(f"  - Added {dataset_name} train split")
            
            if "validation" in splits:
                eval_datasets.append(splits["validation"])
                logger.info(f"  - Added {dataset_name} validation split")
        
        if not train_datasets:
            logger.error("❌ No training datasets found")
            return 1
        
        # Load model and tokenizer
        logger.info("🤖 Loading model and tokenizer...")
        model, tokenizer = model_trainer.load_model_and_tokenizer()
        
        # Setup LoRA
        logger.info("⚙️ Setting up LoRA...")
        model = model_trainer.setup_lora(model)
        
        # Train model
        logger.info("🏋️ Starting model training...")
        trained_model = model_trainer.train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_datasets[0] if len(train_datasets) == 1 else train_datasets,
            eval_dataset=eval_datasets[0] if eval_datasets else None
        )
        
        logger.info("✅ Model training completed")
        
        # Evaluate model
        if not args.skip_evaluation:
            logger.info("📊 Evaluating model...")
            
            # Get test datasets if available
            test_datasets = []
            for dataset_name, dataset_info in datasets.items():
                if "test" in dataset_info["splits"]:
                    test_datasets.append(dataset_info["splits"]["test"])
            
            if test_datasets:
                # Evaluate on test data
                test_dataset = test_datasets[0] if len(test_datasets) == 1 else test_datasets
                metrics = model_evaluator.evaluate_text_generation(
                    model=trained_model,
                    tokenizer=tokenizer,
                    test_dataset=test_dataset,
                    max_samples=args.max_samples
                )
                
                logger.info("📈 Evaluation metrics:")
                for metric, value in metrics.items():
                    logger.info(f"  - {metric}: {value:.4f}")
                
                # Log metrics to MLflow
                model_evaluator.log_metrics_to_mlflow(metrics, args.model_name)
            else:
                logger.warning("⚠️ No test datasets available for evaluation")
        
        # Save model
        logger.info("💾 Saving model...")
        
        # Save locally
        local_path = model_saver.save_locally(
            model=trained_model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            model_name=args.model_name
        )
        logger.info(f"✅ Model saved locally to: {local_path}")
        
        # Save to MLflow
        if not args.skip_mlflow_save:
            logger.info("📊 Saving to MLflow Model Registry...")
            mlflow_uri = model_saver.save_to_mlflow(
                model=trained_model,
                tokenizer=tokenizer,
                model_name=args.model_name,
                metrics=metrics if not args.skip_evaluation else None,
                config=model_trainer.config
            )
            logger.info(f"✅ Model saved to MLflow: {mlflow_uri}")
        
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info(f"📁 Model saved to: {local_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
