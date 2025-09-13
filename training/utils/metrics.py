"""
Metrics computation for model evaluation.
"""

import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for the model.
    
    Args:
        eval_pred: Evaluation predictions from trainer
        
    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_pred
    
    # For causal language modeling
    if len(predictions.shape) == 3:
        # Reshape predictions and labels for token-level evaluation
        predictions = predictions.reshape(-1, predictions.shape[-1])
        labels = labels.reshape(-1)
        
        # Remove padding tokens (-100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
    
    # Compute accuracy
    predicted_ids = np.argmax(predictions, axis=-1)
    accuracy = accuracy_score(labels, predicted_ids)
    
    # Compute precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted_ids, average="weighted", zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
