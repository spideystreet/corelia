"""
Model training service for Mistral-7B fine-tuning with LoRA.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import mlflow
import logging

from ..utils.config import load_training_config, get_env_var, get_env_bool, get_env_int, get_env_float
from ..utils.logging import log_to_mlflow
from ..utils.metrics import compute_metrics


class ModelTrainer:
    """
    Service for training Mistral-7B models with LoRA fine-tuning.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the model trainer.
        
        Args:
            config_path: Path to the training configuration YAML file
        """
        load_training_config()
        self.logger = logging.getLogger("corelia.training.model_trainer")
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.logger.info(f"Loaded configuration from {config_path}")
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow_config = self.config.get("mlflow", {})
        tracking_uri = mlflow_config.get("tracking_uri", "file:./mlruns")
        experiment_name = mlflow_config.get("experiment_name", "corelia-training")
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.logger.info(f"MLflow tracking URI: {tracking_uri}")
        self.logger.info(f"MLflow experiment: {experiment_name}")
    
    def load_model_and_tokenizer(self) -> tuple:
        """
        Load model and tokenizer for training.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_config = self.config["model"]
        model_name = model_config["name"]
        
        self.logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=get_env_var("MODEL_CACHE_DIR", "./models")
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=get_env_var("MODEL_CACHE_DIR", "./models"),
            torch_dtype=getattr(torch, model_config.get("torch_dtype", "float16")),
            device_map=model_config.get("device_map", "auto")
        )
        
        self.logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    
    def setup_lora(self, model) -> Any:
        """
        Setup LoRA configuration for the model.
        
        Args:
            model: Base model to apply LoRA to
            
        Returns:
            Model with LoRA applied
        """
        lora_config = self.config["lora"]
        
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"],
            bias=lora_config["bias"],
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        self.logger.info("LoRA configuration applied successfully")
        return model
    
    def prepare_training_args(self) -> TrainingArguments:
        """
        Prepare training arguments from configuration.
        
        Returns:
            TrainingArguments object
        """
        training_config = self.config["training"]
        
        # Override with environment variables if available
        args = TrainingArguments(
            output_dir=training_config["output_dir"],
            num_train_epochs=get_env_int("NUM_EPOCHS", training_config["num_train_epochs"]),
            per_device_train_batch_size=get_env_int("BATCH_SIZE", training_config["per_device_train_batch_size"]),
            per_device_eval_batch_size=get_env_int("BATCH_SIZE", training_config["per_device_eval_batch_size"]),
            gradient_accumulation_steps=get_env_int("GRADIENT_ACCUMULATION_STEPS", training_config["gradient_accumulation_steps"]),
            learning_rate=get_env_float("LEARNING_RATE", training_config["learning_rate"]),
            weight_decay=training_config["weight_decay"],
            warmup_steps=get_env_int("WARMUP_STEPS", training_config["warmup_steps"]),
            logging_steps=get_env_int("LOGGING_STEPS", training_config["logging_steps"]),
            save_steps=get_env_int("SAVE_STEPS", training_config["save_steps"]),
            eval_steps=training_config["eval_steps"],
            evaluation_strategy=training_config["evaluation_strategy"],
            save_strategy=training_config["save_strategy"],
            load_best_model_at_end=training_config["load_best_model_at_end"],
            metric_for_best_model=training_config["metric_for_best_model"],
            greater_is_better=training_config["greater_is_better"],
            fp16=get_env_bool("FP16", training_config["fp16"]),
            dataloader_num_workers=training_config["dataloader_num_workers"],
            remove_unused_columns=training_config["remove_unused_columns"],
            report_to="mlflow"
        )
        
        return args
    
    def train_model(self, model, tokenizer, train_dataset, eval_dataset=None) -> Any:
        """
        Train the model with the provided datasets.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Trained model
        """
        self.logger.info("Starting model training")
        
        # Prepare training arguments
        training_args = self.prepare_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        # Start training
        with mlflow.start_run():
            # Log configuration
            mlflow.log_params(self.config)
            
            # Train
            trainer.train()
            
            # Save model
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)
            
            self.logger.info("Training completed successfully")
        
        return model
