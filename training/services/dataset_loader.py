"""
Dataset loading service for Mistral-7B fine-tuning datasets.
"""

import os
from typing import Dict, List, Optional, Union
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
import logging

from ..utils.config import load_training_config, get_env_var, get_env_int


class DatasetLoader:
    """
    Service for loading and preprocessing datasets for Mistral-7B fine-tuning.
    """
    
    def __init__(self):
        """Initialize the dataset loader."""
        load_training_config()
        self.logger = logging.getLogger("corelia.training.dataset_loader")
        
        # Setup Hugging Face authentication
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            self.logger.info("Logged in to Hugging Face Hub")
        else:
            self.logger.warning("No HF_TOKEN found, using anonymous access")
    
    def load_dataset_from_hub(
        self, 
        dataset_name: str, 
        split: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> Union[Dataset, DatasetDict]:
        """
        Load a dataset from Hugging Face Hub.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            split: Dataset split to load (train, validation, test)
            max_samples: Maximum number of samples to load (for testing)
            
        Returns:
            Loaded dataset
        """
        try:
            self.logger.info(f"Loading dataset: {dataset_name}")
            
            # Load dataset
            if split:
                dataset = load_dataset(dataset_name, split=split)
            else:
                dataset = load_dataset(dataset_name)
            
            # Limit samples if specified
            if max_samples and hasattr(dataset, '__len__'):
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                    self.logger.info(f"Limited dataset to {max_samples} samples")
            
            self.logger.info(f"Successfully loaded {dataset_name}")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def load_mistral_datasets(self, max_samples: Optional[int] = None) -> Dict[str, Dataset]:
        """
        Load all datasets for Mistral-7B training.
        
        Args:
            max_samples: Maximum samples per dataset (for testing)
            
        Returns:
            Dictionary of loaded datasets with their weights
        """
        datasets_config = {
            "chapin/NACHOS_large": 0.5,
            "Abirate/mediqal": 0.25,
            "alicelacaille/FRASIMED": 0.15
        }
        
        loaded_datasets = {}
        
        for dataset_name, weight in datasets_config.items():
            try:
                dataset = self.load_dataset_from_hub(
                    dataset_name, 
                    split="train",
                    max_samples=max_samples
                )
                loaded_datasets[dataset_name] = {
                    "dataset": dataset,
                    "weight": weight
                }
                self.logger.info(f"Loaded {dataset_name} with weight {weight}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {dataset_name}: {e}")
                # Continue with other datasets
                continue
        
        return loaded_datasets
    
    
    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Union[int, List[str]]]:
        """
        Get information about a dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            "num_samples": len(dataset),
            "features": list(dataset.features.keys()) if hasattr(dataset, 'features') else []
        }
        
        # Get sample data
        if len(dataset) > 0:
            sample = dataset[0]
            info["sample_keys"] = list(sample.keys())
        
        return info
