#!/usr/bin/env python3
"""
Model evaluation script for Mistral-7B fine-tuned models.

This script evaluates a trained model on test datasets and computes
comprehensive metrics including ROUGE, BERTScore, BLEU, and Perplexity.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add training package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.services import DatasetLoader, ModelEvaluator, ModelSaver
from training.utils import setup_logging, load_training_config
from transformers import AutoTokenizer, AutoModelForCausalLM


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Mistral-7B fine-tuned model"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (local path or MLflow model name)"
    )
    
    parser.add_argument(
        "--model-version",
        type=str,
        default="latest",
        help="MLflow model version (if using MLflow model name)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (default: None for full dataset)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file to save evaluation results (JSON format)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def load_model_from_path(model_path: str, model_version: str = "latest"):
    """Load model and tokenizer from path or MLflow."""
    if model_path.startswith("models:/"):
        # Load from MLflow
        model_saver = ModelSaver()
        model, tokenizer = model_saver.load_model_from_mlflow(
            model_name=model_path.replace("models:/", ""),
            version=model_version
        )
    else:
        # Load from local path
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger("corelia.evaluation.main")
    
    logger.info("📊 Starting model evaluation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Max samples: {args.max_samples}")
    
    try:
        # Load training configuration
        load_training_config()
        logger.info("✅ Configuration loaded")
        
        # Initialize services
        logger.info("📊 Initializing services...")
        dataset_loader = DatasetLoader()
        model_evaluator = ModelEvaluator()
        
        # Load model
        logger.info("🤖 Loading model...")
        model, tokenizer = load_model_from_path(args.model_path, args.model_version)
        logger.info("✅ Model loaded successfully")
        
        # Load test datasets
        logger.info("📚 Loading test datasets...")
        datasets = dataset_loader.load_mistral_datasets(
            max_samples=args.max_samples,
            include_test=True
        )
        
        if not datasets:
            logger.error("❌ No datasets loaded successfully")
            return 1
        
        # Collect test datasets
        test_datasets = []
        for dataset_name, dataset_info in datasets.items():
            if "test" in dataset_info["splits"]:
                test_datasets.append(dataset_info["splits"]["test"])
                logger.info(f"  - {dataset_name}: test split loaded")
        
        if not test_datasets:
            logger.error("❌ No test datasets found")
            return 1
        
        # Evaluate model
        logger.info("🔍 Evaluating model...")
        
        # Combine test datasets if multiple
        if len(test_datasets) == 1:
            test_dataset = test_datasets[0]
        else:
            # Concatenate multiple test datasets
            from datasets import concatenate_datasets
            test_dataset = concatenate_datasets(test_datasets)
        
        # Run evaluation
        metrics = model_evaluator.evaluate_text_generation(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            max_samples=args.max_samples
        )
        
        # Compute perplexity
        logger.info("🧮 Computing perplexity...")
        perplexity = model_evaluator.compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset
        )
        metrics["perplexity"] = perplexity
        
        # Display results
        logger.info("📈 Evaluation Results:")
        logger.info("=" * 50)
        
        # ROUGE metrics
        if any(k.startswith("rouge_") for k in metrics.keys()):
            logger.info("🎯 ROUGE Metrics:")
            for metric, value in metrics.items():
                if metric.startswith("rouge_"):
                    logger.info(f"  - {metric}: {value:.4f}")
        
        # BERTScore metrics
        if any(k.startswith("bertscore_") for k in metrics.keys()):
            logger.info("🧠 BERTScore Metrics:")
            for metric, value in metrics.items():
                if metric.startswith("bertscore_"):
                    logger.info(f"  - {metric}: {value:.4f}")
        
        # BLEU score
        if "bleu" in metrics:
            logger.info(f"📝 BLEU Score: {metrics['bleu']:.4f}")
        
        # Perplexity
        logger.info(f"🎲 Perplexity: {metrics['perplexity']:.4f}")
        
        logger.info("=" * 50)
        
        # Log to MLflow
        logger.info("📊 Logging metrics to MLflow...")
        model_evaluator.log_metrics_to_mlflow(metrics, args.model_path)
        
        # Save results to file
        if args.output_file:
            import json
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"💾 Results saved to: {output_path}")
        
        logger.info("🎉 Evaluation completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
