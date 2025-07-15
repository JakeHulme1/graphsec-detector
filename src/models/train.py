import os
import json
import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

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
    # stack the code‐only things
    input_ids      = torch.stack([ex["input_ids"]      for ex in batch], dim=0) # [B, C]
    attention_mask = torch.stack([ex["attention_mask"] for ex in batch], dim=0) # [B, C]
    labels         = torch.stack([ex["labels"]         for ex in batch], dim=0) # [B]

    # stack the graph‐node things
    node_type_ids = torch.stack([ex["node_type_ids"] for ex in batch], dim=0) # [B, N]
    node_mask     = torch.stack([ex["node_mask"]     for ex in batch], dim=0) # [B, N]

    # expand input_ids and attention_mask to hold the nodes
    # just 0‐pad the new token IDs, and use the node_mask as their attention_mask
    pad_ids       = torch.zeros_like(node_type_ids)    # [B, N]
    pad_mask      = node_mask                          # [B, N]

    full_input_ids      = torch.cat([input_ids, pad_ids],     dim=1)  # [B, C+N]
    full_attention_mask = torch.cat([attention_mask,pad_mask],   dim=1)  # [B, C+N]

    # build a position_idx that gives us:
    #  tokens => 2, pads => 1, nodes => 0
    tok_pos       = torch.full_like(attention_mask, 2)   # [B, C]
    pad_pos       = torch.full_like(pad_mask,       1)   # [B, N]
    full_position_idx = torch.cat([tok_pos, pad_pos], dim=1)  # [B, C+N]

    return {
      "input_ids": full_input_ids,
      "attention_mask": full_attention_mask,
      "position_idx": full_position_idx,
      "node_type_ids": node_type_ids,
      "node_mask": node_mask,
      "labels": labels,
    }

    

def train():
    # --- CONFIGURATIONS ---
    # model config
    with open("config/model_config.json") as f:
        mcfg = json.load(f)
    model_cfg = SimpleNamespace(**mcfg)
    
    # training cofig
    with open("config/train_config.yaml") as f:
        train_cfg = yaml.safe_load(f)
    # force to float/int as yaml safe load was parsing as strings
    train_cfg["learning_rate"] = float(train_cfg["learning_rate"])
    train_cfg["weight_decay"]  = float(train_cfg["weight_decay"])
    train_cfg["warmup_steps"]  = int(train_cfg["warmup_steps"])
    train_cfg["batch_size"]    = int(train_cfg["batch_size"])
    train_cfg["epochs"]        = int(train_cfg["epochs"])
    train_cfg["grad_accum_steps"] = int(train_cfg["grad_accum_steps"])

    # set seed & device
    set_seed(train_cfg.get("seed", 42))
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # make sure output dir exists
    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    # tensorboard writer
    writer = SummaryWriter(log_dir=train_cfg["output_dir"])

    #  --- DATASET AND DATALOADER ---
    ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    ds_name = train_cfg["dataset_name"]
    base = os.path.join(ROOT, "datasets", "vudenc", "prepared", ds_name)

    train_path = os.path.join(base, "train.jsonl")
    val_path = os.path.join(base, "val.jsonl")
    test_path = os.path.join(base, "test.jsonl")

    train_ds = VulnerabilityDataset(train_path,
                                    max_seq_len=model_cfg.max_seq_length,
                                    max_nodes=model_cfg.max_nodes)
    if len(train_ds) == 0:
        raise RuntimeError(f"Found 0 records in {train_path!r}.  "
                       f"Check that this file exists and is non-empty.")
    val_ds = VulnerabilityDataset(val_path,
                                  max_seq_len=model_cfg.max_seq_length,
                                max_nodes=model_cfg.max_nodes)
    test_ds = VulnerabilityDataset(test_path,
                                   max_seq_len=model_cfg.max_seq_length,
                                max_nodes=model_cfg.max_nodes)
    
    train_loader = DataLoader(train_ds,
                              batch_size=train_cfg["batch_size"],
                              shuffle=True,
                              num_workers=train_cfg.get("num_workers", 4),
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds,
                            batch_size=train_cfg["batch_size"],
                            shuffle=False,
                            num_workers=train_cfg.get("num_workers", 4),
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_ds,
                             batch_size=train_cfg["batch_size"],
                             shuffle=False,
                             num_workers=train_cfg.get("num_workers", 4),
                             collate_fn=collate_fn)
    
    # --- MODEL, OPTIMISER, SCHEDULER ---
    # load model
    classifier = GCBertClassifier(model_cfg).to(device)
    # separate parameters for decay
    no_decay = ["bias", "LayerNorm.weight"] # very sensitive parameters
    optim_groups = [
        {
            "params": [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": train_cfg["weight_decay"]
        },
        {
            "params": [p for n,p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optim_groups, lr=train_cfg["learning_rate"])

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

            # ─── ZERO‐PAD THE GRAPH‐NODE PART TO MATCH ATTENTION_MASK
            bs, L = batch["attention_mask"].shape     # e.g. [B, 384]
            _,  N = batch["node_mask"].shape          # e.g. [B, 16]
            if N != L:
                pad_mask = torch.zeros(bs, L-N,
                    device=batch["node_mask"].device,
                    dtype=batch["node_mask"].dtype)
                batch["node_mask"]     = torch.cat([batch["node_mask"],     pad_mask], dim=1)
                pad_type = torch.zeros(bs, L-N,
                    device=batch["node_type_ids"].device,
                    dtype=batch["node_type_ids"].dtype)
                batch["node_type_ids"] = torch.cat([batch["node_type_ids"], pad_type], dim=1)
            
            loss, _ = classifier(**batch)
            loss = loss / train_cfg["grad_accum_steps"]
            loss.backward()
            running_loss += loss.item()

            if step % train_cfg["grad_accum_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        avg_train_loss = running_loss / len(train_loader)
        histories["train_loss"].append(avg_train_loss)

        # --- train set metrcis ---
        running_train_loss = 0.0
        classifier.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                bs, L = batch["attention_mask"].shape
                _,  N = batch["node_mask"].shape
                if N != L:
                    pad_mask = torch.zeros(bs, L-N,
                        device=batch["node_mask"].device,
                        dtype=batch["node_mask"].dtype)
                    batch["node_mask"]     = torch.cat([batch["node_mask"],     pad_mask], dim=1)
                    pad_type = torch.zeros(bs, L-N,
                        device=batch["node_type_ids"].device,
                        dtype=batch["node_type_ids"].dtype)
                    batch["node_type_ids"] = torch.cat([batch["node_type_ids"], pad_type], dim=1)

                
                loss, logits = classifier(**batch)
                running_train_loss += loss.item()
                all_logits.append(logits)
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

                bs, L = batch["attention_mask"].shape
                _,  N = batch["node_mask"].shape
                if N != L:
                    pad_mask = torch.zeros(bs, L-N,
                        device=batch["node_mask"].device,
                        dtype=batch["node_mask"].dtype)
                    batch["node_mask"]     = torch.cat([batch["node_mask"],     pad_mask], dim=1)
                    pad_type = torch.zeros(bs, L-N,
                        device=batch["node_type_ids"].device,
                        dtype=batch["node_type_ids"].dtype)
                    batch["node_type_ids"] = torch.cat([batch["node_type_ids"], pad_type], dim=1)


                loss, logits = classifier(**batch)
                running_val_loss += loss.item()
                all_logits.append(logits)
                all_labels.append(batch["labels"])
        avg_val_loss = running_val_loss / len(val_loader)
        histories["val_loss"].append(avg_val_loss)

        val_logits = torch.cat(all_logits, dim=0)
        val_labels = torch.cat(all_labels, dim=0)
        val_metrics = compute_metrics(val_logits, val_labels)

        # ************** DEBUGING **************
        import matplotlib.pyplot as plt
        import random

        # Convert logits to probabilities
        probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()

        # Plot histogram of predicted probabilities
        plt.hist(probs, bins=20, range=(0, 1))
        plt.title("Validation Set: Predicted Positive Class Probabilities")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.savefig(os.path.join(train_cfg["output_dir"], f"val_prob_hist_epoch_{epoch}.png"))
        plt.close()

        # Print 10 random examples
        indices = random.sample(range(len(probs)), 10)
        for idx in indices:
            print(f"Example {idx}: Prob={probs[idx]:.4f}, Label={val_labels[idx].item()}")
        # ************** DEBUGGING **************


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
        

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("ROC_AUC/val", val_metrics["roc_auc"], epoch)

        # Log all metrics
        for name, val in train_metrics.items():
            writer.add_scalar(f"Metric/train/{name}", val, epoch)
        for name, val in val_metrics.items():
            writer.add_scalar(f"Metric/val/{name}", val, epoch)


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

            bs, L = batch["attention_mask"].shape
            _,  N = batch["node_mask"].shape
            if N != L:
                pad_mask = torch.zeros(bs, L-N,
                    device=batch["node_mask"].device,
                    dtype=batch["node_mask"].dtype)
                batch["node_mask"]     = torch.cat([batch["node_mask"],     pad_mask], dim=1)
                pad_type = torch.zeros(bs, L-N,
                    device=batch["node_type_ids"].device,
                    dtype=batch["node_type_ids"].dtype)
                batch["node_type_ids"] = torch.cat([batch["node_type_ids"], pad_type], dim=1)

            _, logits = classifier(**batch)
            all_logits.append(logits)
            all_labels.append(batch["labels"])
    test_logits = torch.cat(all_logits, dim=0)
    test_labels = torch.cat(all_labels, dim=0)
    test_metrics = compute_metrics(test_logits, test_labels)
    print("Test metrics:", test_metrics)

    writer.close()

    overfit_one_batch(classifier, train_loader, device)

# ************** DEBUGING **************
def overfit_one_batch(classifier, train_loader, device):
    classifier.train()
    batch = next(iter(train_loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer = AdamW(classifier.parameters(), lr=1e-3)
    loss_history = []

    for i in range(100):
        optimizer.zero_grad()

        bs, L = batch["attention_mask"].shape
        _, N = batch["node_mask"].shape
        if N != L:
            pad_mask = torch.zeros(bs, L - N,
                                   device=batch["node_mask"].device,
                                   dtype=batch["node_mask"].dtype)
            batch["node_mask"] = torch.cat([batch["node_mask"], pad_mask], dim=1)
            pad_type = torch.zeros(bs, L - N,
                                   device=batch["node_type_ids"].device,
                                   dtype=batch["node_type_ids"].dtype)
            batch["node_type_ids"] = torch.cat([batch["node_type_ids"], pad_type], dim=1)

        loss, _ = classifier(**batch)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")

    print(f"Final loss after 100 iterations: {loss_history[-1]}")

# ************** DEBUGING **************

if __name__ == "__main__":
    train()