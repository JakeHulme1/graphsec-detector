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


def train(train_model: bool = True):
    # Load configs
    with open("config/model_config.json") as f:
        mcfg = json.load(f)
    model_cfg = SimpleNamespace(**mcfg)

    cfg_path = os.getenv("TRAIN_CONFIG_PATH", "config/train_config.yaml")
    with open(cfg_path) as f:
        train_cfg = yaml.safe_load(f)

    # Hyperparams
    lr            = float(train_cfg["learning_rate"])
    weight_decay  = float(train_cfg["weight_decay"])
    warmup_scale  = float(train_cfg.get("warmup_steps_scale", 0.1))
    batch_size    = int(train_cfg["batch_size"])
    epochs        = int(train_cfg["epochs"])
    grad_acc      = int(train_cfg["grad_accum_steps"])

    set_seed(train_cfg.get("seed", 42))
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Prepare output
    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    # Load data
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base = os.path.join(ROOT, "datasets", "vudenc", "prepared", train_cfg["dataset_name"])
    train_ds = VulnerabilityDataset(os.path.join(base, "train.jsonl"), max_seq_len=model_cfg.max_seq_length, max_nodes=model_cfg.max_nodes)
    val_ds   = VulnerabilityDataset(os.path.join(base, "val.jsonl"),   max_seq_len=model_cfg.max_seq_length, max_nodes=model_cfg.max_nodes)
    test_ds  = VulnerabilityDataset(os.path.join(base, "test.jsonl"),  max_seq_len=model_cfg.max_seq_length, max_nodes=model_cfg.max_nodes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=train_cfg.get("num_workers",4), collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=train_cfg.get("num_workers",4), collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=train_cfg.get("num_workers",4), collate_fn=collate_fn)

    # Loss function
    all_train_labels = torch.cat([ex["labels"].unsqueeze(0) for ex in train_ds])
    num_pos = all_train_labels.sum().item()
    num_neg = len(all_train_labels) - num_pos
    class_weights = torch.tensor([1.0, num_neg/num_pos], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"Train pos %: {100*num_pos/len(all_train_labels):.2f}")
    print(f"Val pos %:   {100*sum(ex['labels'] for ex in val_ds)/len(val_ds):.2f}")
    print(f"Test pos %:  {100*sum(ex['labels'] for ex in test_ds)/len(test_ds):.2f}")

    # Model
    classifier = GCBertClassifier(model_cfg).to(device)

    # Freeze all but head
    head = []
    for name, param in classifier.named_parameters():
        if name.startswith("classifier."):
            param.requires_grad = True
            head.append((name, param))
        else:
            param.requires_grad = False

    # Optimizer on head only
    no_decay = ["bias", "LayerNorm.weight"]
    head_nonbias = [p for n,p in head if not any(nd in n for nd in no_decay)]
    head_bias    = [p for n,p in head if     any(nd in n for nd in no_decay)]
    optimizer = AdamW([
        {"params": head_nonbias, "lr": lr, "weight_decay": weight_decay},
        {"params": head_bias,    "lr": lr, "weight_decay": 0.0},
    ])

    # Scheduler
    total_steps = (len(train_loader)//grad_acc) * epochs
    warmup_steps = int(warmup_scale * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Histories
    histories = {"train_loss":[], "val_loss":[]}
    for m in ["accuracy","precision","recall","f1","pr_auc","roc_auc"]:
        histories[f"train_{m}"] = []
        histories[f"val_{m}"]   = []

    best_val_loss = float("inf")
    early_stop    = 0

    # Summary file path (across experiments)
    summary_path = os.path.join(os.path.dirname(train_cfg["output_dir"]), "experiment_summary.txt")

    if train_model:
        for epoch in range(1, epochs+1):
            classifier.train()
            total_train_loss = 0
            for step, batch in enumerate(train_loader, start=1):
                batch = {k:v.to(device) for k,v in batch.items()}
                logits = classifier(**batch)
                loss   = loss_fn(logits.view(-1,2), batch["labels"].view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                if step % grad_acc == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                total_train_loss += loss.item()
            histories["train_loss"].append(total_train_loss/len(train_loader))

            # Validation
            classifier.eval()
            val_loss = 0
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k:v.to(device) for k,v in batch.items()}
                    logits = classifier(**batch)
                    val_loss += loss_fn(logits.view(-1,2), batch["labels"].view(-1)).item()
                    all_logits.append(logits)
                    all_labels.append(batch["labels"])
            avg_val_loss = val_loss/len(val_loader)
            histories["val_loss"].append(avg_val_loss)
            metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
            for k,v in metrics.items(): histories[f"val_{k}"].append(v)

            # Checkpoint & early-stop
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(classifier.state_dict(), os.path.join(train_cfg["output_dir"], "best.pt"))
                early_stop = 0
            else:
                early_stop += 1

            line = f"Epoch {epoch} | train_loss={histories['train_loss'][-1]:.4f} "
            line += f"val_loss={histories['val_loss'][-1]:.4f} "
            line += f"val_pr_auc={histories['val_pr_auc'][-1]:.4f}\n"
            print(line.strip())
            # Append to summary file
            with open(summary_path, "a") as sf:
                if epoch == 1:
                    sf.write(f"Experiment: {train_cfg['output_dir']}\n")
                sf.write(line)

            if early_stop >= 3:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save plot
                # Save plot of only losses to avoid dimension mismatch
        try:
            loss_hist = {"train_loss": histories["train_loss"],
                         "val_loss":   histories["val_loss"]}
            plot_training(loss_hist,
                          os.path.join(train_cfg["output_dir"], "training_plot.png"))
        except Exception as e:
            print(f"Warning: could not save training_plot.png: {e}")(histories, os.path.join(train_cfg["output_dir"], "training_plot.png"))
        except OSError as e:
            print(f"Warning: could not save plot: {e}")

    else:
        print("Skipping training, loading best checkpoint…")
        checkpoint = os.path.join(train_cfg["output_dir"], "best.pt")
        classifier.load_state_dict(torch.load(checkpoint, map_location=device))
        classifier.eval()

    # Final test evaluation
    all_logits, all_labels, test_loss = [], [], 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            logits = classifier(**batch)
            test_loss += loss_fn(logits.view(-1,2), batch["labels"].view(-1)).item()
            all_logits.append(logits)
            all_labels.append(batch["labels"])
    avg_test_loss = test_loss/len(test_loader)
    print(f"Test loss: {avg_test_loss:.4f}")

    # Write final test metrics to summary
    test_metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    with open(summary_path, "a") as sf:
        sf.write(f"Test loss: {avg_test_loss:.4f}\n")
        sf.write(f"Test metrics: {test_metrics}\n\n")

    # ROC & PR curves
    fpr, tpr, _ = roc_curve(torch.cat(all_labels).cpu().numpy(), F.softmax(torch.cat(all_logits), dim=1)[:,1].cpu().numpy())
    roc_auc      = auc(fpr, tpr)
    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"ROC Curve (AUC={roc_auc:.2f})")
    plt.savefig(os.path.join(train_cfg["output_dir"], "test_roc_curve.png"))

    precision, recall, _ = precision_recall_curve(torch.cat(all_labels).cpu().numpy(), F.softmax(torch.cat(all_logits), dim=1)[:,1].cpu().numpy())
    pr_auc              = auc(recall, precision)
    plt.figure()
    PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc).plot()
    plt.title(f"Precision-Recall (AUC={pr_auc:.2f})")
    plt.savefig(os.path.join(train_cfg["output_dir"], "test_pr_curve.png"))

    # Threshold sweep print
    print("\nThreshold sweep:")
    for thr in [0.5, 0.4, 0.3, 0.2, 0.1]:
        preds = (F.softmax(torch.cat(all_logits), dim=1)[:,1].cpu().numpy() >= thr).astype(int)
        true  = torch.cat(all_labels).cpu().numpy()
        prec, rec, f1, _ = precision_recall_fscore_support(true, preds, average="binary", zero_division=0)
        line = f"Thr {thr:.1f} | Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}\n"
        print(line.strip())
        with open(summary_path, "a") as sf:
            sf.write(line)

    print("Test metrics:", test_metrics)
    with open(summary_path, "a") as sf:
        sf.write("---\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or evaluate GraphCodeBERT classifier")
    parser.add_argument(
        "--train_model", type=lambda x: x.lower() in ("1","true","yes"), default=True,
        help="Whether to run training. Pass false to skip training and only eval."
    )
    args = parser.parse_args()
    train(train_model=args.train_model)
