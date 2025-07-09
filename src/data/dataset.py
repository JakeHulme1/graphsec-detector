import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

REL2ID= {
    "comesFrom": 0,
    "computedFrom": 1,
}

class VulnerabilityDataset(Dataset):
    def __init__(self, path, model_name="microsoft/graphcodebert-base", max_seq_len=128, max_nodes=128):
        """
    Convert prepared data set into PyTorch tensors and return fields GraphCodeBERT expects. 

        Args:
            path (str): path to a JSONL file (one record per line)
            model_name (str): pretrained GraphCodeBERT tokenizer
            max_seq_len (int): max tokens for code sequence
            max_nodes (int): max number of graph nodes
        """
        self.records = []
        with open(path, 'r') as f:
            for line in f:
                self.records.append(json.loads(line))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_len = max_seq_len
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]

        #  1. Tokenize code
        enc = self.tokenizer(
            rec["code"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc.input_ids.squeeze(0)  # (max_seq_len,)
        attention_mask = enc.attention_mask.squeeze(0)   # (max_seq_len,)

        #  2. Node tpyes and mask
        rels = [n[2] for n in rec["graph_nodes"]]
        type_ids = [REL2ID[r] for r in rels]
        node_count = len(type_ids)
        pad_len = self.max_nodes - node_count
        node_type_ids = torch.tensor(type_ids + [0]*pad_len, dtype=torch.long)  # (max_nodes,)
        node_mask     = torch.tensor([1]*node_count + [0]*pad_len, dtype=torch.long)  # (max_nodes,)

        #  3. Edge index
        #  keep only edges whose nodes are within the real node count
        edges = [(u, v) for u, v in rec["graph_edges"] if u < node_count and v < node_count]
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()  # (2, E)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        #  4. Label
        label = torch.tensor(rec["label"], dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "node_type_ids":  node_type_ids,
            "node_mask":      node_mask,
            "edge_index":     edge_index,
            "labels":         label,
        }