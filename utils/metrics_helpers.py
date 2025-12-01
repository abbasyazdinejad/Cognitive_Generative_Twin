import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary'
) -> Dict[str, float]:
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def compute_anomaly_threshold(
    scores: np.ndarray,
    percentile: float = 95.0
) -> float:
    
    threshold = np.percentile(scores, percentile)
    return float(threshold)


def compute_anomaly_predictions(
    scores: np.ndarray,
    threshold: Optional[float] = None,
    percentile: float = 95.0
) -> np.ndarray:
    
    if threshold is None:
        threshold = compute_anomaly_threshold(scores, percentile)
    
    predictions = (scores > threshold).astype(int)
    
    return predictions


def normalize_scores(
    scores: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    
    if method == 'minmax':
        min_val = np.min(scores)
        max_val = np.max(scores)
        
        if max_val - min_val < 1e-8:
            return np.zeros_like(scores)
        
        normalized = (scores - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std < 1e-8:
            return np.zeros_like(scores)
        
        normalized = (scores - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def compute_reconstruction_scores(
    model,
    data: torch.Tensor,
    device: torch.device,
    reduction: str = 'mean'
) -> np.ndarray:
    
    model.eval()
    
    scores = []
    
    with torch.no_grad():
        data = data.to(device)
        
        if hasattr(model, 'reconstruction_error'):
            errors = model.reconstruction_error(data, reduction=reduction)
            scores = errors.cpu().numpy()
        else:
            recon = model(data)
            
            if reduction == 'mean':
                errors = torch.mean((data - recon) ** 2, dim=(1, 2))
            elif reduction == 'sum':
                errors = torch.sum((data - recon) ** 2, dim=(1, 2))
            else:
                errors = (data - recon) ** 2
            
            scores = errors.cpu().numpy()
    
    return scores


def batch_to_device(batch: Dict, device: torch.device) -> Dict:
    
    batch_device = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        else:
            batch_device[key] = value
    
    return batch_device


def count_parameters(model: torch.nn.Module) -> int:
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_numpy_dict(data: Dict, path: str):
    
    np.savez(path, **data)


def load_numpy_dict(path: str) -> Dict:
    
    data = np.load(path, allow_pickle=True)
    
    return {key: data[key] for key in data.files}