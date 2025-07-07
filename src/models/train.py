import os
import json
import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from types import SimpleNamespace

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

def train():
    # --- CONFIGURATIONS ---
    # model config
    with open("config/model_config.json") as f:
        mcfg = json.load(f)
    model_cfg = SimpleNamespace(**mcfg)
    
    # training cofig
    with open("config/train_config.yaml") as f:
        train_cfg = yaml.safe_load(f)

    # set seed & device
    set_seed(train_cfg.get("seed", 42))
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # make sure output dir exists
    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    #  --- DATASET AND DATALOADER ---
    ROOT = os.getcwd()
    ds_name = train_cfg["dataset_name"]
    base = os.path.join(ROOT, "datasets", "vudenc", "prepared", ds_name)

    train_path = os.path.join(base, "train.jsonl")
    val_path = os.path.join(base, "val.jsonl")
    test_path = os.path.join(base, "test.jsonl")

    train_ds = VulnerabilityDataset(train_path,
                                    max_seq_len=model_cfg.max_seq_length,
                                    max_nodes=model_cfg.max_nodes)
    val_ds = VulnerabilityDataset(val_path,
                                  max_seq_len=model_cfg.max_seq_length,
                                max_nodes=model_cfg.max_nodes)
    test_ds = VulnerabilityDataset(test_path,
                                   max_seq_len=model_cfg.max_seq_length,
                                max_nodes=model_cfg.max_nodes)
    
    train_loader = DataLoader(train_ds,
                              batch_size=train_cfg["batch_size"],
                              shuffle=True,
                              num_workers=train_cfg.get("num_workers", 4))
    val_loader = DataLoader(val_ds,
                            batch_size=train_cfg["batch_size"],
                            shuffle=False,
                            num_workers=train_cfg.get("num_workers", 4))
    test_loader = DataLoader(test_ds,
                             batch_size=train_cfg["batch_size"],
                             shuffle=False,
                             num_workers=train_cfg.get("num_workers", 4))
    
    # --- MODEL, OPTIMISER, SCHEDULER ---
    # load model
    classifier = GCBertClassifier(model_cfg).model.to(train_cfg.device)
    
    # separate parameters for decay
    no_decay = ["bias", "LayerNorm.weight"] # very sensitive parameters
    optim_groups = [
        {
            "params": [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": train_cfg.weight_decay
        },
        {
            "params": [p for n,p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optim_groups, lr=train_cfg.learning_rate)

    total_steps = len(train_loader) // train_cfg["grad_accum_steps"] * train_cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        train_cfg["warmup_steps"], 
        total_steps
    )

    # --- HISTORIES AND BEST MODEL TRACKING ---
    histories = {
        "train_loss": [],
        "val_loss":   [],
        # train metrics
        **{f"train_{m}": [] for m in ["accuracy","precision","recall","f1","pr_auc","roc_auc"]},
        # val metrics
        **{f"val_{m}": [] for m in ["accuracy","precision","recall","f1","pr_auc","roc_auc"]},
    }
    best_val_roc = -float("inf")

    # --- EPOCH LOOP ---
    for epoch in range(1, train_cfg["epochs"] + 1):
        # training
        classifier.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = classifier(**batch)
            loss = out.loss / train_cfg["grad_accum_steps"]
            loss.backward()
            running_loss += loss.item()

            if step % train_cfg["grad_accum_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        avg_train_loss = running_loss / len(train_loader)
        histories["train_loss"].append(avg_train_loss)

        # --- train set metrcis ---
        classifier.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = classifier(**batch)
                running_val_loss += out.loss.item()
                all_logits.append(out.logits)
                all_labels.append(batch["labels"])
        train_logits = torch.cat(all_logits, dim=0)
        train_labels = torch.cat(all_labels, dim=0)
        train_metrics = compute_metrics(train_logits, train_labels)
        for name, val in train_metrics.items():
            histories[f"train_{name}"].append(val)

        # --- validation set metrics ---
        running_val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = classifier(**batch)
                running_val_loss += out.loss.item()
                all_logits.append(out.logits)
                all_labels.append(batch["labels"])
        avg_val_loss = running_val_loss / len(val_loader)
        histories["val_loss"].append(avg_val_loss)

        val_logits = torch.cat(all_logits, dim=0)
        val_labels = torch.cat(all_labels, dim=0)
        val_metrics = compute_metrics(val_logits, val_labels)
        for name, val in val_metrics.items():
            histories[f"val_{name}"].append(val)

        # save best model by ROC-AUC
        if val_metrics["roc_auc"] > best_val_roc:
            best_val_roc = val_metrics["roc_auc"]
            torch.save(
                classifier.state_dict(),
                os.path.join(train_cfg["output_dir"], "best.pt")
            )

        print(f"Epoch {epoch:2d} | "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss: .4f} "
            f"val_roc_auc={val_metrics['roc_auc']:.4f}")
        
    # --- PLOT ALL CURVES ---
    plot_training(
        histories,
        os.path.join(train_cfg["output_dir"], "training_plot.png")
    )

    # --- FINAL TEST SET EVALUATION ---
    classifier.load_state_dict(torch.load(os.path.join(train_cfg["output_dir"], "best.pt")))
    classifier.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = classifier(**batch)
            all_logits.append(out.logits)
            all_labels.append(batch["labels"])
    test_logits = torch.cat(all_logits, dim=0)
    test_labels = torch.cat(all_labels, dim=0)
    test_metrics = compute_metrics(test_logits, test_labels)
    print("Test metrics:", test_metrics)

if __name__ == "__main__":
    train()