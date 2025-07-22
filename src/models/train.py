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
from torch.utils.tensorboard import SummaryWriter
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
    # stack the code‐only tensors
    input_ids      = torch.stack([ex["input_ids"]      for ex in batch], dim=0)
    attention_mask = torch.stack([ex["attention_mask"] for ex in batch], dim=0)
    labels         = torch.stack([ex["labels"]         for ex in batch], dim=0)

    # stack the graph‐node tensors
    node_type_ids = torch.stack([ex["node_type_ids"] for ex in batch], dim=0)
    node_mask     = torch.stack([ex["node_mask"]     for ex in batch], dim=0)

    # zero‐pad the new token IDs and use node_mask as their attention
    pad_ids  = torch.zeros_like(node_type_ids)
    pad_mask = node_mask

    full_input_ids      = torch.cat([input_ids,      pad_ids],  dim=1)
    full_attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

    # build position idx: tokens→their index+2, pads→1, nodes→0
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


def train():
    # ─── LOAD CONFIGS ─────────────────────────────────────────────────────────────
    with open("config/model_config.json") as f:
        mcfg = json.load(f)
    model_cfg = SimpleNamespace(**mcfg)

    # get current train config for hyper parameter sweep
    cfg_path = os.getenv("TRAIN_CONFIG_PATH", "config/train_config.yaml")
    with open(cfg_path) as f:
        train_cfg = yaml.safe_load(f)
    # cast to correct types
    train_cfg["learning_rate"]   = float(train_cfg["learning_rate"])
    train_cfg["weight_decay"]    = float(train_cfg["weight_decay"])
    train_cfg["warmup_steps"]    = int(train_cfg["warmup_steps"])
    train_cfg["batch_size"]      = int(train_cfg["batch_size"])
    train_cfg["epochs"]          = int(train_cfg["epochs"])
    train_cfg["grad_accum_steps"] = int(train_cfg["grad_accum_steps"])

    set_seed(train_cfg.get("seed", 42))
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    os.makedirs(train_cfg["output_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=train_cfg["output_dir"])

    # ─── LOAD DATA ────────────────────────────────────────────────────────────────
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base = os.path.join(ROOT, "datasets", "vudenc", "prepared", train_cfg["dataset_name"])
    train_ds = VulnerabilityDataset(os.path.join(base, "train.jsonl"),
                                    max_seq_len=model_cfg.max_seq_length,
                                    max_nodes=model_cfg.max_nodes)
    val_ds   = VulnerabilityDataset(os.path.join(base,   "val.jsonl"),
                                    max_seq_len=model_cfg.max_seq_length,
                                    max_nodes=model_cfg.max_nodes)
    test_ds  = VulnerabilityDataset(os.path.join(base,  "test.jsonl"),
                                    max_seq_len=model_cfg.max_seq_length,
                                    max_nodes=model_cfg.max_nodes)

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True,  num_workers=train_cfg.get("num_workers",4),
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"],
                              shuffle=False, num_workers=train_cfg.get("num_workers",4),
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=train_cfg["batch_size"],
                              shuffle=False, num_workers=train_cfg.get("num_workers",4),
                              collate_fn=collate_fn)

    # ─── CLASS‐WEIGHTED LOSS ──────────────────────────────────────────────────────
    all_train_labels = torch.cat([ex["labels"].unsqueeze(0) for ex in train_ds])
    num_pos = all_train_labels.sum().item()
    num_neg = len(all_train_labels) - num_pos
    class_weights = torch.tensor([1.0, num_neg / num_pos], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"Train pos %: {100*num_pos/len(all_train_labels):.2f}")
    print(f"Val pos %:   {100*sum(ex['labels'] for ex in val_ds)/len(val_ds):.2f}")
    print(f"Test pos %:  {100*sum(ex['labels'] for ex in test_ds)/len(test_ds):.2f}")

    # ─── MODEL, OPTIM, SCHED ─────────────────────────────────────────────────────
    classifier = GCBertClassifier(model_cfg).to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    optim_groups = [
        {"params": [p for n,p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": train_cfg["weight_decay"]},
        {"params": [p for n,p in classifier.named_parameters() if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=train_cfg["learning_rate"])
    total_steps = (len(train_loader)//train_cfg["grad_accum_steps"])*train_cfg["epochs"]
    scheduler   = get_linear_schedule_with_warmup(optimizer,
                                                  train_cfg["warmup_steps"],
                                                  total_steps)

    histories = {"train_loss":[], "val_loss":[]}
    for m in ["accuracy","precision","recall","f1","pr_auc","roc_auc"]:
        histories[f"train_{m}"] = []
        histories[f"val_{m}"]   = []

    best_val_roc = -1
    early_stop   = 0

    # ─── EPOCH LOOP ──────────────────────────────────────────────────────────────
    for epoch in range(1, train_cfg["epochs"]+1):
        # — TRAINING —
        classifier.train()
        total_train_loss = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k:v.to(device) for k,v in batch.items()}

            # pad graph‐node portion if needed
            bs, L = batch["attention_mask"].shape
            _,  N = batch["node_mask"].shape
            if N!=L:
                pad_kwargs = dict(device=batch["node_mask"].device,
                                  dtype=batch["node_mask"].dtype)
                batch["node_mask"]     = torch.cat([batch["node_mask"],
                                                    torch.zeros(bs,L-N,**pad_kwargs)],dim=1)
                pad_kwargs["dtype"] = batch["node_type_ids"].dtype
                batch["node_type_ids"] = torch.cat([batch["node_type_ids"],
                                                    torch.zeros(bs,L-N,**pad_kwargs)],dim=1)

            logits = classifier(**batch)
            loss   = loss_fn(logits.view(-1,2), batch["labels"].view(-1))
            loss = loss / train_cfg["grad_accum_steps"]
            loss.backward()
            total_train_loss += loss.item()

            if step % train_cfg["grad_accum_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss/len(train_loader)
        histories["train_loss"].append(avg_train_loss)

        # — EVAL (train / val) —
        for phase, loader in [("train",train_loader),("val",val_loader)]:
            classifier.eval()
            all_logits, all_labels, running_loss = [], [], 0
            with torch.no_grad():
                for batch in loader:
                    batch = {k:v.to(device) for k,v in batch.items()}
                    bs, L = batch["attention_mask"].shape
                    _,  N = batch["node_mask"].shape
                    if N!=L:
                        pad_kwargs = dict(device=batch["node_mask"].device,
                                          dtype=batch["node_mask"].dtype)
                        batch["node_mask"]     = torch.cat([batch["node_mask"],
                                                            torch.zeros(bs,L-N,**pad_kwargs)],dim=1)
                        pad_kwargs["dtype"] = batch["node_type_ids"].dtype
                        batch["node_type_ids"] = torch.cat([batch["node_type_ids"],
                                                            torch.zeros(bs,L-N,**pad_kwargs)],dim=1)

                    logits = classifier(**batch)
                    loss   = loss_fn(logits.view(-1,2), batch["labels"].view(-1))
                    running_loss += loss.item()
                    all_logits.append(logits)
                    all_labels.append(batch["labels"])

            avg_loss = running_loss/len(loader)
            histories[f"{phase}_loss"].append(avg_loss)
            metrics = compute_metrics(torch.cat(all_logits,dim=0),
                                      torch.cat(all_labels,dim=0))
            for k,v in metrics.items():
                histories[f"{phase}_{k}"].append(v)

            if phase=="val":
                # checkpoint by ROC
                if metrics["roc_auc"]>best_val_roc:
                    best_val_roc=metrics["roc_auc"]
                    torch.save(classifier.state_dict(),
                               os.path.join(train_cfg["output_dir"],"best.pt"))
                    early_stop=0
                else:
                    early_stop+=1

        print(f"Epoch {epoch:2d} | "
              f"train_loss={avg_train_loss:.4f} "
              f"val_loss={histories['val_loss'][-1]:.4f} "
              f"val_roc_auc={histories['val_roc_auc'][-1]:.4f}")

        writer.add_scalar("Loss/train", histories["train_loss"][-1], epoch)
        writer.add_scalar("Loss/val", histories["val_loss"][-1], epoch)
        writer.add_scalar("ROC_AUC/val", histories["val_roc_auc"][-1], epoch)
        for name in ["accuracy","precision","recall","f1","pr_auc","roc_auc"]:
            writer.add_scalar(f"Metric/train/{name}", histories[f"train_{name}"][-1], epoch)
            writer.add_scalar(f"Metric/val/{name}",   histories[f"val_{name}"][-1],   epoch)

        if early_stop>=3:
            print(f"Early stopping at epoch {epoch}")
            break

    # ─── FINAL PLOTS & TEST EVAL ─────────────────────────────────────────────────
    plot_training(histories, os.path.join(train_cfg["output_dir"],"training_plot.png"))

    # load best and eval on test
    classifier.load_state_dict(torch.load(os.path.join(train_cfg["output_dir"],"best.pt")))
    classifier.eval()
    all_logits, all_labels, test_loss = [], [], 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            bs, L = batch["attention_mask"].shape
            _,  N = batch["node_mask"].shape
            if N!=L:
                pad_kwargs = dict(device=batch["node_mask"].device,
                                  dtype=batch["node_mask"].dtype)
                batch["node_mask"]     = torch.cat([batch["node_mask"],
                                                    torch.zeros(bs,L-N,**pad_kwargs)],dim=1)
                pad_kwargs["dtype"] = batch["node_type_ids"].dtype
                batch["node_type_ids"] = torch.cat([batch["node_type_ids"],
                                                    torch.zeros(bs,L-N,**pad_kwargs)],dim=1)

            logits = classifier(**batch)
            loss   = loss_fn(logits.view(-1,2), batch["labels"].view(-1))
            test_loss+=loss.item()
            all_logits.append(logits)
            all_labels.append(batch["labels"])

    avg_test_loss = test_loss/len(test_loader)
    print(f"Test loss: {avg_test_loss:.4f}")

    test_logits = torch.cat(all_logits, dim=0)
    test_labels = torch.cat(all_labels, dim=0)
    probs = F.softmax(test_logits, dim=1)[:,1].cpu().numpy()
    true  = test_labels.cpu().numpy()

    # ROC curve
    fpr, tpr, _ = roc_curve(true, probs)
    roc_auc      = auc(fpr, tpr)
    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"ROC Curve (AUC={roc_auc:.2f})")
    plt.savefig(os.path.join(train_cfg["output_dir"],"test_roc_curve.png"))

    # PR curve
    precision, recall, _ = precision_recall_curve(true, probs)
    pr_auc              = auc(recall, precision)
    plt.figure()
    PrecisionRecallDisplay(precision=precision, recall=recall,
                           average_precision=pr_auc).plot()
    plt.title(f"Precision-Recall (AUC={pr_auc:.2f})")
    plt.savefig(os.path.join(train_cfg["output_dir"],"test_pr_curve.png"))

    # threshold sweep
    print("\nThreshold sweep:")
    for thr in [0.5,0.4,0.3, 0.2, 0.1]:
        preds = (probs>=thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(true, preds,
                                                           average="binary",
                                                           zero_division=0)
        print(f"Thr {thr:.1f} | Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}")

    print("Test metrics:", compute_metrics(test_logits, test_labels))
    writer.close()


if __name__ == "__main__":
    train()
