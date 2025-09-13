"""
Dataset loading service for Mistral-7B fine-tuning datasets.
"""

import os
from typing import Dict, List, Optional, Union
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
import logging

from ..utils.config import load_training_config, get_env_var, get_env_int, get_env_dict, get_env_float


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
    
    def get_dataset_splits(self, dataset_name: str) -> List[str]:
        """
        Get available splits for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of available splits
        """
        try:
            dataset_info = load_dataset(dataset_name, split=None)
            splits = list(dataset_info.keys())
            self.logger.info(f"Available splits for {dataset_name}: {splits}")
            return splits
        except Exception as e:
            self.logger.error(f"Failed to get splits for {dataset_name}: {e}")
            return ["train"]  # Default fallback
    
    def load_mistral_datasets(
        self, 
        max_samples: Optional[int] = None,
        include_test: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load all datasets for Mistral-7B training with proper splits.
        
        Args:
            max_samples: Maximum samples per dataset (for testing)
            include_test: Whether to load test splits for evaluation
            
        Returns:
            Dictionary of loaded datasets with their weights and splits
        """
        # Get datasets configuration from environment
        dataset_names = get_env_list("DATASETS_MISTRAL")
        
        # Create datasets config with individual weights
        datasets_config = {}
        for dataset_name in dataset_names:
            # Map dataset names to their weight variables
            if "NACHOS" in dataset_name:
                weight = get_env_float("DATASET_WEIGHT_NACHOS")
            elif "mediqal" in dataset_name.lower():
                weight = get_env_float("DATASET_WEIGHT_MEDIQAL")
            elif "FRASIMED" in dataset_name:
                weight = get_env_float("DATASET_WEIGHT_FRASIMED")
            else:
                # Default weight for unknown datasets
                weight = 1.0
                self.logger.warning(f"Unknown dataset {dataset_name}, using default weight 1.0")
            
            datasets_config[dataset_name] = weight
        
        loaded_datasets = {}
        
        for dataset_name, weight in datasets_config.items():
            try:
                # Get available splits
                splits = self.get_dataset_splits(dataset_name)
                
                dataset_info = {
                    "weight": weight,
                    "splits": {}
                }
                
                # Load train split
                if "train" in splits:
                    train_dataset = self.load_dataset_from_hub(
                        dataset_name, 
                        split="train",
                        max_samples=max_samples
                    )
                    dataset_info["splits"]["train"] = train_dataset
                
                # Load validation split
                if "validation" in splits or "val" in splits:
                    val_split = "validation" if "validation" in splits else "val"
                    val_dataset = self.load_dataset_from_hub(
                        dataset_name, 
                        split=val_split,
                        max_samples=max_samples
                    )
                    dataset_info["splits"]["validation"] = val_dataset
                
                # Load test split if requested
                if include_test and "test" in splits:
                    test_dataset = self.load_dataset_from_hub(
                        dataset_name, 
                        split="test",
                        max_samples=max_samples
                    )
                    dataset_info["splits"]["test"] = test_dataset
                
                loaded_datasets[dataset_name] = dataset_info
                self.logger.info(f"Loaded {dataset_name} with weight {weight} and splits: {list(dataset_info['splits'].keys())}")
                
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
