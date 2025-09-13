"""
Model evaluation service for Mistral-7B text generation.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datasets import Dataset
import logging
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM
import mlflow

from ..utils.config import load_training_config, get_env_var, get_env_list
from ..utils.logging import log_to_mlflow


class ModelEvaluator:
    """
    Service for evaluating Mistral-7B text generation models.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        load_training_config()
        self.logger = logging.getLogger("corelia.training.evaluator")
        
        # Get enabled metrics from environment
        self.enabled_metrics = get_env_list("EVALUATION_METRICS")
        
        # Initialize ROUGE scorer if needed
        if "rouge" in self.enabled_metrics:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize BLEU smoothing function if needed
        if "bleu" in self.enabled_metrics:
            self.bleu_smoothing = SmoothingFunction().method1
        
        self.logger.info(f"Model evaluator initialized with metrics: {self.enabled_metrics}")
    
    def evaluate_text_generation(
        self, 
        model: Any, 
        tokenizer: Any, 
        test_dataset: Dataset,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate text generation model (Mistral-7B) using ROUGE and BERTScore.
        
        Args:
            model: Trained model
            tokenizer: Model tokenizer
            test_dataset: Test dataset
            max_samples: Maximum samples to evaluate (for testing)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Starting text generation evaluation")
        
        # Limit samples if specified
        if max_samples and len(test_dataset) > max_samples:
            test_dataset = test_dataset.select(range(max_samples))
            self.logger.info(f"Limited evaluation to {max_samples} samples")
        
        # Prepare data
        predictions = []
        references = []
        
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(test_dataset):
                if i % 10 == 0:
                    self.logger.info(f"Evaluating sample {i}/{len(test_dataset)}")
                
                # Get input text (assuming 'text' field exists)
                input_text = sample.get('text', '')
                if not input_text:
                    continue
                
                # Generate prediction
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                
                # Move to device
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = generated_text[len(input_text):].strip()
                
                predictions.append(prediction)
                references.append(sample.get('target', input_text))  # Use target if available
        
        # Calculate metrics
        metrics = self._calculate_generation_metrics(predictions, references)
        
        self.logger.info("Text generation evaluation completed")
        return metrics
    
    
    def _calculate_generation_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate generation metrics based on enabled metrics."""
        metrics = {}
        
        # ROUGE metrics
        if "rouge" in self.enabled_metrics:
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                for metric in rouge_scores.keys():
                    rouge_scores[metric].append(scores[metric].fmeasure)
            
            # Average ROUGE scores
            for metric, scores in rouge_scores.items():
                metrics[f'rouge_{metric}'] = np.mean(scores)
        
        # BERTScore
        if "bertscore" in self.enabled_metrics:
            try:
                P, R, F1 = bert_score(predictions, references, lang="fr", verbose=False)
                metrics['bertscore_precision'] = P.mean().item()
                metrics['bertscore_recall'] = R.mean().item()
                metrics['bertscore_f1'] = F1.mean().item()
            except Exception as e:
                self.logger.warning(f"BERTScore calculation failed: {e}")
                metrics['bertscore_precision'] = 0.0
                metrics['bertscore_recall'] = 0.0
                metrics['bertscore_f1'] = 0.0
        
        # BLEU score
        if "bleu" in self.enabled_metrics:
            bleu_scores = []
            for pred, ref in zip(predictions, references):
                # Tokenize for BLEU calculation
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.bleu_smoothing)
                    bleu_scores.append(bleu)
            
            if bleu_scores:
                metrics['bleu'] = np.mean(bleu_scores)
            else:
                metrics['bleu'] = 0.0
        
        return metrics
    
    
    def log_metrics_to_mlflow(self, metrics: Dict[str, float], model_name: str) -> None:
        """Log evaluation metrics to MLflow."""
        try:
            with mlflow.start_run():
                mlflow.log_metrics(metrics)
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("evaluation_type", "model_evaluation")
                
            self.logger.info(f"Logged {len(metrics)} metrics to MLflow")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics to MLflow: {e}")
    
    def compute_perplexity(self, model: Any, tokenizer: Any, test_dataset: Dataset) -> float:
        """
        Compute perplexity for language modeling evaluation.
        
        Args:
            model: Trained model
            tokenizer: Model tokenizer
            test_dataset: Test dataset
            
        Returns:
            Perplexity score
        """
        self.logger.info("Computing perplexity")
        
        total_loss = 0.0
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for sample in test_dataset:
                text = sample.get('text', '')
                if not text:
                    continue
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs['input_ids'].numel()
                total_tokens += inputs['input_ids'].numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.logger.info(f"Perplexity: {perplexity:.4f}")
        return perplexity
