import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Calculate: accuracy, precision, recall, F1, PR-AUC, ROC-AUC
    """
    assert isinstance(logits, torch.Tensor), "logits must be a torch.Tensor"
    assert isinstance(labels, torch.Tensor), "labels must be a torch.Tensor"
    assert labels.dtype == torch.long, f"labels must be torch.long, got {labels.dtype}"
    assert logits.dtype in (torch.float16, torch.float32, torch.float64), f"logits must be a floating dtype, got {logits.dtype}"
    
    # probability of positive class - move to NumPy
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    # predicted labels 
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

    # ground truth lables - move to NumPy
    true = labels.detach().cpu().numpy()

    # compute metrics
    acc = accuracy_score(true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, preds, average='binary', zero_division=0
    )
    pr_auc = average_precision_score(true, probs)
    roc_auc = roc_auc_score(true, probs)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }