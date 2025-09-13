#!/usr/bin/env python3
"""
Model deployment script for pushing trained models to Hugging Face Hub.

This script takes a trained model and deploys it to Hugging Face Hub
with proper model cards and metadata.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add training package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.services import ModelSaver
from training.utils import setup_logging, load_training_config
from transformers import AutoTokenizer, AutoModelForCausalLM


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Deploy Mistral-7B fine-tuned model to Hugging Face Hub"
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
        "--hf-model-name",
        type=str,
        required=True,
        help="Name for the model on Hugging Face Hub (e.g., 'username/model-name')"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the model private on Hugging Face Hub"
    )
    
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload fine-tuned Mistral-7B for French medical text processing",
        help="Commit message for the model upload"
    )
    
    parser.add_argument(
        "--create-repo",
        action="store_true",
        default=True,
        help="Create the repository if it doesn't exist"
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
    """Main deployment function."""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger("corelia.deployment.main")
    
    logger.info("🚀 Starting model deployment to Hugging Face Hub")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"HF model name: {args.hf_model_name}")
    logger.info(f"Private: {args.private}")
    
    try:
        # Load training configuration
        load_training_config()
        logger.info("✅ Configuration loaded")
        
        # Check HF token
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("❌ HF_TOKEN environment variable not set")
            logger.error("Please set your Hugging Face token:")
            logger.error("export HF_TOKEN=your_token_here")
            return 1
        
        # Initialize services
        logger.info("📊 Initializing services...")
        model_saver = ModelSaver()
        
        # Load model
        logger.info("🤖 Loading model...")
        model, tokenizer = load_model_from_path(args.model_path, args.model_version)
        logger.info("✅ Model loaded successfully")
        
        # Create repository if requested
        if args.create_repo:
            logger.info("📁 Creating Hugging Face repository...")
            repo_url = model_saver.create_hf_repo(
                model_name=args.hf_model_name,
                private=args.private
            )
            if repo_url:
                logger.info(f"✅ Repository created: {repo_url}")
            else:
                logger.warning("⚠️ Repository creation failed or already exists")
        
        # Deploy to Hugging Face Hub
        logger.info("📤 Deploying model to Hugging Face Hub...")
        model_url = model_saver.save_to_huggingface_hub(
            model=model,
            tokenizer=tokenizer,
            model_name=args.hf_model_name,
            commit_message=args.commit_message,
            private=args.private,
            create_repo=not args.create_repo  # Don't create if we already did
        )
        
        if model_url:
            logger.info("🎉 Model deployed successfully!")
            logger.info(f"📱 Model URL: {model_url}")
            logger.info(f"🔗 Direct link: https://huggingface.co/{args.hf_model_name}")
            
            # Display usage instructions
            logger.info("\n📖 Usage Instructions:")
            logger.info("=" * 50)
            logger.info("```python")
            logger.info("from transformers import AutoTokenizer, AutoModelForCausalLM")
            logger.info("")
            logger.info(f"tokenizer = AutoTokenizer.from_pretrained('{args.hf_model_name}')")
            logger.info(f"model = AutoModelForCausalLM.from_pretrained('{args.hf_model_name}')")
            logger.info("")
            logger.info("# Example usage")
            logger.info('text = "Patient présentant des symptômes de..."')
            logger.info("inputs = tokenizer(text, return_tensors='pt')")
            logger.info("outputs = model.generate(**inputs, max_length=100)")
            logger.info("result = tokenizer.decode(outputs[0], skip_special_tokens=True)")
            logger.info("```")
            logger.info("=" * 50)
            
        else:
            logger.error("❌ Model deployment failed")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
