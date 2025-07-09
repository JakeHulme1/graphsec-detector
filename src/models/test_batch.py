import json
from types import SimpleNamespace
from data.dataset import VulnerabilityDataset
from models.graphcodebert_cls import GCBertClassifier
import torch

# Load model config
with open("config/model_config.json") as f:
    cfg_dict = json.load(f)
cfg_dict["device"] = "cpu"  # test on CPU
cfg = SimpleNamespace(**cfg_dict)

# Load dataset
ds = VulnerabilityDataset("datasets/vudenc/prepared/command_injection/train.jsonl",
                          max_seq_len=cfg.max_seq_length,
                          max_nodes=cfg.max_nodes)

# Batchify one sample
sample = ds[0]
batch = {k: v.unsqueeze(0) for k, v in sample.items()}

# Initialise model
model = GCBertClassifier(cfg).to("cpu")
model.eval()

# Forward pass
with torch.no_grad():
    output = model(**batch)
    if isinstance(output, tuple):
        loss, logits = output
    else:
        loss, logits = output.loss, output.logits

print("Test batch passed!")
print("Loss:", loss)
print("Logits:", logits)