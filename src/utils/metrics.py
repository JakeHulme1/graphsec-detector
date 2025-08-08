# utils/metrics.py
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float | None = 0.5,  # None => use argmax (old behavior)
) -> dict:
    """
    Calculate accuracy, precision, recall, F1 at the given threshold (if provided),
    plus threshold-free PR-AUC (AP) and ROC-AUC from probabilities.
    """
    assert isinstance(logits, torch.Tensor), "logits must be a torch.Tensor"
    assert isinstance(labels, torch.Tensor), "labels must be a torch.Tensor"
    assert labels.dtype == torch.long, f"labels must be torch.long, got {labels.dtype}"
    assert logits.ndim == 2 and logits.size(1) == 2, f"logits must be (N,2), got {tuple(logits.shape)}"
    assert logits.dtype in (torch.float16, torch.float32, torch.float64), f"logits must be float dtype"

    # probs of positive class
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    true  = labels.detach().cpu().numpy().astype(np.int64)

    if threshold is None:
        # old behavior: argmax over logits
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
    else:
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        preds = (probs >= float(threshold)).astype(np.int64)

    acc = accuracy_score(true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, preds, average="binary", zero_division=0
    )
    pr_auc = average_precision_score(true, probs)  # PR-AUC (AP)
    roc_auc = roc_auc_score(true, probs)_
