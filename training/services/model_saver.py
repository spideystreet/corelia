"""
Model saving service for Mistral-7B models to Hugging Face Hub and MLflow.
"""

import os
import shutil
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import mlflow
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from ..utils.config import load_training_config, get_env_var
from ..utils.logging import log_to_mlflow


class ModelSaver:
    """
    Service for saving Mistral-7B models to Hugging Face Hub and MLflow.
    """
    
    def __init__(self):
        """Initialize the model saver."""
        load_training_config()
        self.logger = logging.getLogger("corelia.training.model_saver")
        
        # Setup Hugging Face authentication
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            self.api = HfApi()
            self.logger.info("Logged in to Hugging Face Hub")
        else:
            self.logger.warning("No HF_TOKEN found, cannot upload to Hub")
            self.api = None
    
    def save_to_huggingface_hub(
        self, 
        model: Any, 
        tokenizer: Any, 
        model_name: str,
        commit_message: str = "Upload trained model",
        private: bool = True
    ) -> Optional[str]:
        """
        Save model and tokenizer to Hugging Face Hub.
        
        Args:
            model: Trained model
            tokenizer: Model tokenizer
            model_name: Name for the model on HF Hub
            commit_message: Git commit message
            private: Whether to make the model private
            
        Returns:
            Model URL if successful, None otherwise
        """
        if not self.api:
            self.logger.error("No HF_TOKEN available, cannot upload to Hub")
            return None
        
        try:
            self.logger.info(f"Saving model to Hugging Face Hub: {model_name}")
            
            # Create temporary directory
            temp_dir = Path(f"./temp_{model_name}")
            temp_dir.mkdir(exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            # Create model card
            self._create_model_card(temp_dir, model_name)
            
            # Upload to Hub
            repo_url = self.api.create_repo(
                repo_id=model_name,
                private=private,
                exist_ok=True
            )
            
            self.api.upload_folder(
                folder_path=temp_dir,
                repo_id=model_name,
                commit_message=commit_message
            )
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            self.logger.info(f"Model successfully uploaded to: {repo_url}")
            return repo_url
            
        except Exception as e:
            self.logger.error(f"Failed to upload model to Hub: {e}")
            return None
    
    def save_to_mlflow(
        self, 
        model: Any, 
        tokenizer: Any, 
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model to MLflow Model Registry.
        
        Args:
            model: Trained model
            tokenizer: Model tokenizer
            model_name: Name for the model
            metrics: Evaluation metrics
            config: Model configuration
            
        Returns:
            Model URI
        """
        try:
            self.logger.info(f"Saving model to MLflow: {model_name}")
            
            # Create temporary directory
            temp_dir = Path(f"./temp_mlflow_{model_name}")
            temp_dir.mkdir(exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            # Save configuration
            if config:
                import yaml
                with open(temp_dir / "config.yaml", 'w') as f:
                    yaml.dump(config, f)
            
            # Log model to MLflow
            with mlflow.start_run():
                # Log metrics if provided
                if metrics:
                    mlflow.log_metrics(metrics)
                
                # Log configuration
                if config:
                    mlflow.log_params(config)
                
                # Log model
                model_uri = mlflow.log_artifacts(str(temp_dir), artifact_path="model")
                
                # Register model
                mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name
                )
                
                self.logger.info(f"Model registered in MLflow: {model_name}")
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            return model_uri
            
        except Exception as e:
            self.logger.error(f"Failed to save model to MLflow: {e}")
            raise
    
    def save_locally(
        self, 
        model: Any, 
        tokenizer: Any, 
        output_dir: str,
        model_name: str
    ) -> str:
        """
        Save model locally with proper structure.
        
        Args:
            model: Trained model
            tokenizer: Model tokenizer
            output_dir: Output directory
            model_name: Model name
            
        Returns:
            Path to saved model
        """
        try:
            self.logger.info(f"Saving model locally: {model_name}")
            
            # Create output directory
            model_dir = Path(output_dir) / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            # Create model card
            self._create_model_card(model_dir, model_name)
            
            # Save model info
            model_info = {
                "model_name": model_name,
                "model_type": "causal_lm",
                "base_model": "mistralai/Mistral-7B-v0.1",
                "framework": "pytorch",
                "language": "french",
                "domain": "medical",
                "fine_tuning": "lora"
            }
            
            import json
            with open(model_dir / "model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
            
            self.logger.info(f"Model saved locally to: {model_dir}")
            return str(model_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to save model locally: {e}")
            raise
    
    def _create_model_card(self, model_dir: Path, model_name: str) -> None:
        """Create a model card for the saved model."""
        model_card = f"""---
language: fr
license: apache-2.0
tags:
- medical
- french
- healthcare
- corelia
- mistral
- lora
---

# {model_name}

## Model Description

This model is a LoRA fine-tuned version of Mistral-7B for French medical text processing, developed as part of the Corelia project.

## Training Data

The model was fine-tuned on the following French medical datasets:
- NACHOS (50% weight)
- MediQAl (25% weight) 
- FRASIMED (15% weight)

## Fine-tuning Method

- **Base Model**: mistralai/Mistral-7B-v0.1
- **Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj
- **Rank**: 16
- **Alpha**: 32

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{model_name}")
model = AutoModelForCausalLM.from_pretrained("{model_name}")

# Example usage
text = "Patient présentant des symptômes de..."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Performance

This model is designed for French medical text processing and should be evaluated on relevant medical tasks.

## Limitations

- This model is a proof-of-concept and should not be used for medical administrative purposes
- Performance may vary on different medical domains
- Requires proper evaluation before production use

## Citation

```bibtex
@software{{corelia2024,
  title={{Corelia: French Medical AI for Healthcare}},
  author={{Corelia Team}},
  year={{2024}},
  url={{https://github.com/spideystreet/corelia}}
}}
```
"""
        
        with open(model_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(model_card)
    
    def load_model_from_mlflow(self, model_name: str, version: str = "latest") -> tuple:
        """
        Load model from MLflow Model Registry.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            self.logger.info(f"Loading model from MLflow: {model_name}")
            
            # Load model from MLflow
            model_uri = f"models:/{model_name}/{version}"
            
            # Download model artifacts
            model_path = mlflow.artifacts.download_artifacts(model_uri)
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            self.logger.info("Model loaded successfully from MLflow")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load model from MLflow: {e}")
            raise
