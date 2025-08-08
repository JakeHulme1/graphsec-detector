#!/usr/bin/env python
import os
import json
import yaml
import random
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from types import SimpleNamespace
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_recall_fscore_support
)

from data.dataset import VulnerabilityDataset
from models.graphcodebert_cls import GCBertClassifier
from utils.metrics import compute_metrics
from utils.plot import plot_training

from torch import nn

# ─── SMOOTHING HELPERS ────────────────────────────────────────────
def smooth_np(x: list[float], window: int = 5) -> list[float]:
    """
    Moving average via NumPy convolution.
    Pads edges so output length == input length.
    """
    if len(x) < window or window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same").tolist()

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps   = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=-1).clamp(self.eps, 1-self.eps)
        pt    = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        w     = self.alpha[targets]
        loss  = - w * (1 - pt)**self.gamma * pt.log()
        return loss.mean()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    input_ids      = torch.stack([ex["input_ids"]      for ex in batch], dim=0)
    attention_mask = torch.stack([ex["attention_mask"] for ex in batch], dim=0)
    labels         = torch.stack([ex["labels"]         for ex in batch], dim=0)

    node_type_ids = torch.stack([ex["node_type_ids"] for ex in batch], dim=0)
    node_mask     = torch.stack([ex["node_mask"]     for ex in batch], dim=0)

    pad_ids  = torch.zeros_like(node_type_ids)
    pad_mask = node_mask

    full_input_ids      = torch.cat([input_ids,      pad_ids],  dim=1)
    full_attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

    bs, seq_len = input_ids.shape
    tok_pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bs, -1) + 2
    tok_pos = tok_pos * attention_mask + (1 - attention_mask)
    node_pos = torch.zeros_like(node_mask)
    node_pos[node_mask == 0] = 1
    full_position_idx = torch.cat([tok_pos, node_pos], dim=1)

    return {
        "input_ids":      full_input_ids,
        "attention_mask": full_attention_mask,
        "position_idx":   full_position_idx,
        "node_type_ids":  node_type_ids,
        "node_mask":      node_mask,
        "labels":         labels,
    }


def _local_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float):
    """Fallback if utils.metrics.compute_metrics doesn't accept `threshold`."""
    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    y     = labels.detach().cpu().numpy()
    preds = (probs >= threshold).astype(np.int64)

    acc = (preds == y).mean().item()
    p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)

    pr_prec, pr_rec, _ = precision_recall_curve(y, probs)
    pr_auc_val = auc(pr_rec, pr_prec)
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc_val = auc(fpr, tpr)
    return {
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        "pr_auc": pr_auc_val, "roc_auc": roc_auc_val
    }


def _safe_compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float):
    try:
        # If your utils.metrics.compute_metrics supports `threshold`
        return compute_metrics(logits, labels, threshold=threshold)  # type: ignore[arg-type]
    except TypeError:
        # Backward-compatible path
        return _local_metrics(logits, labels, threshold)


def train(train_model: bool = True):
    # ─── LOAD CONFIGS ─────────────────────────────────────────────────────────────
    cfg_path = os.getenv("TRAIN_CONFIG_PATH", "config/train_config.yaml")
    with open(cfg_path) as f:
        train_cfg = yaml.safe_load(f)
    with open("config/model_config.json") as f:
        mcfg = json.load(f)
    mcfg["classifier_dropout"] = float(
        train_cfg.get("classifier_dropout",
                      mcfg.get("classifier_dropout", 0.1))
    )
    model_cfg = SimpleNamespace(**mcfg)

    # ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────
    lr           = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg["weight_decay"])
    warmup_scale = float(train_cfg.get("warmup_steps_scale", 0.1))
    batch_size   = int(train_cfg["batch_size"])
    epochs       = int(train_cfg["epochs"])
    grad_acc     = int(train_cfg["grad_accum_steps"])
    thr          = float(train_cfg.get("threshold", 0.5))
    smooth_win   = int(train_cfg.get("smoothing_window", 5))  # 1 disables smoothing (for plots only)

    set_seed(train_cfg.get("seed", 42))
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    os.makedirs(train_cfg["output_dir"], exist_ok=True)
    summary_path = os.path.join(os.path.dirname(train_cfg["output_dir"]), "experiment_summary.txt")

    # ─── DATA LOADERS ─────────────────────────────────────────────────────────────
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base = os.path.join(ROOT, "datasets", "vudenc", "prepared", train_cfg["dataset_name"])
    train_ds = VulnerabilityDataset(os.path.join(base, "train.jsonl"),
                                    max_seq_len=model_cfg.max_seq_length,
                                    max_nodes=model_cfg.max_nodes)
    val_ds   = VulnerabilityDataset(os.path.join(base, "val.jsonl"),
                                    max_seq_len=model_cfg.max_seq_length,
                                    max_nodes=model_cfg.max_nodes)
    test_ds  = VulnerabilityDataset(os.path.join(base, "test.jsonl"),
                                    max_seq_len=model_cfg.max_seq_length,
                                    max_nodes=model_cfg.max_nodes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=train_cfg.get("num_workers",4),
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=train_cfg.get("num_workers",4),
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=train_cfg.get("num_workers",4),
                              collate_fn=collate_fn)

    # ─── LOSS FUNCTION ────────────────────────────────────────────────────────────
    all_train_labels = torch.cat([ex["labels"].unsqueeze(0) for ex in train_ds])
    num_pos = all_train_labels.sum().item()
    num_neg = len(all_train_labels) - num_pos
    class_weights = torch.tensor([1.0, num_neg/num_pos], device=device)
    if train_cfg.get("loss", "") == "focal":
        gamma = float(train_cfg.get("focal_gamma", 2.0))
        loss_fn = WeightedFocalLoss(alpha=class_weights, gamma=gamma)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Train pos %: {100*num_pos/len(all_train_labels):.2f}")
    print(f"Val pos %:   {100*sum(ex['labels'] for ex in val_ds)/len(val_ds):.2f}")
    print(f"Test pos %:  {100*sum(ex['labels'] for ex in test_ds)/len(test_ds):.2f}")

    # ─── MODEL & FREEZE/UNFREEZE ─────────────────────────────────────────────────
    classifier = GCBertClassifier(model_cfg).to(device)

    # Unfreeze only the head + last transformer block
    head, last_block = [], []
    for name, param in classifier.named_parameters():
        if name.startswith("encoder.encoder.layer.10.") or name.startswith("encoder.encoder.layer.11.") or name.startswith("encoder.encoder.layer.9."):
            param.requires_grad = True
            last_block.append((name, param))
        elif name.startswith("classifier.") or name.startswith("dense") or name.startswith("dropout"):
            param.requires_grad = True
            head.append((name, param))
        else:
            param.requires_grad = False

    # ─── OPTIMIZER & SCHEDULER ───────────────────────────────────────────────────
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = AdamW([
        {"params": [p for n,p in head if not any(nd in n for nd in no_decay)],
         "lr": lr, "weight_decay": weight_decay},
        {"params": [p for n,p in head if     any(nd in n for nd in no_decay)],
         "lr": lr, "weight_decay": 0.0},
        {"params": [p for n,p in last_block if not any(nd in n for nd in no_decay)],
         "lr": lr * 0.2, "weight_decay": weight_decay},
        {"params": [p for n,p in last_block if     any(nd in n for nd in no_decay)],
         "lr": lr * 0.2, "weight_decay": 0.0},
    ])

    total_steps  = (len(train_loader)//grad_acc) * epochs
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    # ─── TRAINING LOOP ──────────────────────────────────────────────────────────
    histories = {"train_loss":[], "val_loss":[]}
    for m in ["accuracy","precision","recall","f1","pr_auc","roc_auc"]:
        histories[f"train_{m}"] = []
        histories[f"val_{m}"]   = []

    best_val_pr   = -float("inf")
    early_stop    = 0

    if train_model:
        for epoch in range(1, epochs+1):
            classifier.train()
            total_train_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                batch = {k:v.to(device) for k,v in batch.items()}
                logits = classifier(**batch)
                loss   = loss_fn(logits.view(-1,2), batch["labels"].view(-1))
                loss.backward()
                if step % 100 == 0:
                    print(f"[step {step}] avg grad = {classifier.classifier.weight.grad.abs().mean():.3e}")

                if step % grad_acc == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                total_train_loss += loss.item()

            histories["train_loss"].append(total_train_loss / len(train_loader))

            # compute and record training metrics (thresholded)
            classifier.eval()
            all_train_logits, all_train_labels = [], []
            with torch.no_grad():
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logits = classifier(**batch)
                    all_train_logits.append(logits)
                    all_train_labels.append(batch["labels"])
            train_metrics = _safe_compute_metrics(
                torch.cat(all_train_logits),
                torch.cat(all_train_labels),
                threshold=thr
            )
            for name, value in train_metrics.items():
                histories[f"train_{name}"].append(value)
            classifier.train()

            # validation (thresholded for metrics; raw logits used for loss)
            classifier.eval()
            val_loss, all_logits, all_labels = 0.0, [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k:v.to(device) for k,v in batch.items()}
                    logits = classifier(**batch)
                    val_loss += loss_fn(logits.view(-1,2), batch["labels"].view(-1)).item()
                    all_logits.append(logits)
                    all_labels.append(batch["labels"])

            avg_val_loss = val_loss / len(val_loader)
            histories["val_loss"].append(avg_val_loss)
            metrics = _safe_compute_metrics(torch.cat(all_logits), torch.cat(all_labels), threshold=thr)
            for k,v in metrics.items():
                histories[f"val_{k}"].append(v)

            # Step scheduler on RAW PR-AUC (no smoothing bias)
            scheduler.step(metrics["pr_auc"])

            # checkpoint & early-stop on RAW PR-AUC
            if metrics["pr_auc"] > best_val_pr:
                best_val_pr = metrics["pr_auc"]
                torch.save(classifier.state_dict(),
                           os.path.join(train_cfg["output_dir"], "best.pt"))
                early_stop = 0
            else:
                early_stop += 1

            line = (f"Epoch {epoch} | train_loss={histories['train_loss'][-1]:.4f} "
                    f"val_loss={histories['val_loss'][-1]:.4f} "
                    f"val_pr_auc={histories['val_pr_auc'][-1]:.4f}\n")
            print(line.strip())
            with open(summary_path, "a") as sf:
                if epoch == 1:
                    sf.write(f"Experiment: {train_cfg['output_dir']} (thr={thr}, smooth_win={smooth_win})\n")
                sf.write(line)

            if early_stop >= 8:
                print(f"Early stopping at epoch {epoch}")
                break

        # ─── SAVE TRAIN/VAL PLOTS ────────────────────────────────────────
        try:
            metrics_png = os.path.join(train_cfg["output_dir"], f"metrics_thr-{thr}.png")
            sm_hist = {k: smooth_np(v, window=smooth_win) for k, v in histories.items()}
            plot_training(sm_hist, metrics_png)

            plt.figure()
            epochs_range = list(range(1, len(histories["train_loss"]) + 1))
            plt.plot(epochs_range, histories["train_loss"], label="train_loss")
            plt.plot(epochs_range, histories["val_loss"],   label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(train_cfg["output_dir"], "training_plot.png"))
            plt.close()
        except Exception as e:
            print(f"Warning: could not save training_plot.png: {e}")
            print(f"Warning: could not save metrics.png: {e}")

    else:
        print("Skipping training, loading best checkpoint…")
        checkpoint = os.path.join(train_cfg["output_dir"], "best.pt")
        classifier.load_state_dict(torch.load(checkpoint, map_location=device))
        classifier.eval()

    # ─── FINAL TEST EVALUATION ─────────────────────────────────────────────
    all_logits, all_labels, test_loss = [], [], 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            logits = classifier(**batch)
            test_loss += loss_fn(logits.view(-1,2), batch["labels"].view(-1)).item()
            all_logits.append(logits)
            all_labels.append(batch["labels"])

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test loss: {avg_test_loss:.4f}")

    test_metrics = _safe_compute_metrics(torch.cat(all_logits), torch.cat(all_labels), threshold=thr)
    with open(summary_path, "a") as sf:
        sf.write(f"Test loss: {avg_test_loss:.4f}\n")
        sf.write(f"Test metrics (thr={thr}): {test_metrics}\n\n")

    # ─── ROC & PR CURVES (threshold-free) ────────────────────────────────────
    probs = F.softmax(torch.cat(all_logits), dim=1)[:,1].cpu().numpy()
    true  = torch.cat(all_labels).cpu().numpy()

    fpr, tpr, _ = roc_curve(true, probs)
    roc_auc_val  = auc(fpr, tpr)
    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_val).plot()
    plt.title(f"ROC Curve (AUC={roc_auc_val:.2f})")
    plt.savefig(os.path.join(train_cfg["output_dir"], "test_roc_curve.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(true, probs)
    pr_auc_val          = auc(recall, precision)
    plt.figure()
    PrecisionRecallDisplay(precision=precision, recall=recall,
                           average_precision=pr_auc_val).plot()
    plt.title(f"Precision-Recall (AUC={pr_auc_val:.2f})")
    plt.savefig(os.path.join(train_cfg["output_dir"], "test_pr_curve.png"))
    plt.close()

    # ─── THRESHOLD SWEEP (for reporting) ───────────────────────────────────
    print("\nThreshold sweep:")
    for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
        preds = (probs >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(true, preds,
                                                           average="binary",
                                                           zero_division=0)
        line = f"Thr {t:.1f} | Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}\n"
        print(line.strip())
        with open(summary_path, "a") as sf:
            sf.write(line)

    with open(os.path.join(train_cfg["output_dir"], "chosen_threshold.txt"), "w") as f:
        f.write(f"{thr:.4f}\n")

    print("Test metrics:", test_metrics)
    with open(summary_path, "a") as sf:
        sf.write("---\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or evaluate GraphCodeBERT classifier")
    parser.add_argument(
        "--train_model",
        type=lambda x: x.lower() in ("1","true","yes"),
        default=True,
        help="Whether to run training. Pass false to skip training and only eval."
    )
    args = parser.parse_args()
    train(train_model=args.train_model)
