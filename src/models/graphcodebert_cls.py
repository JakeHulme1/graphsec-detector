import os
import sys
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

GRAPH_CODE_BERT_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", "extern", "GraphCodeBERT"))
if GRAPH_CODE_BERT_ROOT not in sys.path:
    sys.path.insert(0, GRAPH_CODE_BERT_ROOT)

from codesearch.model import Model as GraphCodeBERTEncoder

class GCBertClassifier(nn.Module):
    """
    A classifier on top on the GrpahCodeBERT encoder in codesearch/model.py
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Load and override cofif from HuggingFace 
        config = AutoConfig.from_pretrained(
            cfg.model_name_or_path,
            hidden_dropout_prob=getattr(cfg, "hidden_dropout_prob", None),
            classifier_dropout=getattr(cfg, "classifier_dropout", None),
        )

        # INstantiate the raw ROberta (GraphCodeBERT) encoder itself
        # Use AutoModel here to get the base transformer, not a sequence classification head
        hf_encoder = AutoModel.from_pretrained(
            cfg.model_name_or_path,
            config=config,
        )

        # Wrap it with the graph-aware embedding logic
        self.encoder = GraphCodeBERTEncoder(hf_encoder)

        # # Dropout
        # self.dropout = nn.Dropout(config.classifier_dropout or config.hidden_dropout_prob or 0.1)

        # # Classification head: just a single layer from hidden_size -> num_labels
        # self.classifier = nn.Linear(config.hidden_size, cfg.num_labels)

        # smaller‐dropout + two‐layer MLP head for more capacity
        hidden_size = config.hidden_size
        mid_size    = hidden_size // 2
        self.dropout1   = nn.Dropout(0.2)
        self.dense      = nn.Linear(hidden_size, mid_size)
        self.dropout2   = nn.Dropout(0.2)
        self.classifier = nn.Linear(mid_size, cfg.num_labels)


    def forward(self, **batch):
        """
        Accept the keys the prepared dataset produces:
          input_ids, attention_mask,
          node_type_ids, node_mask, edge_index, labels

        Only need:
          code_inputs=input_ids
          attn_mask=attention_mask
          position_idx=node_type_ids
          labels=labels (optional)
        """
        # pull them out (will KeyError if something missing)
        code_inputs     = batch["input_ids"]
        attn_mask       = batch["attention_mask"]
        labels          = batch.get("labels", None)

        # Get position_idx if present, otherwise treat everything as token (no nodes)
        if "position_idx" in batch:
            position_idx = batch["position_idx"]
        else:
            position_idx = torch.full_like(code_inputs, 2)

        # Run graph‐aware encoder -> pooled [batch, hidden_size]
        pooled = self.encoder(
            code_inputs=code_inputs,
            attn_mask=attn_mask,
            position_idx=position_idx,
            nl_inputs=None,
        )

        # x = self.dropout(pooled)
        # logits = self.classifier(x)

        x = self.dropout1(pooled)
        x = torch.relu(self.dense(x))
        x = self.dropout2(x)
        logits = self.classifier(x)

        # # Cross‐entropy loss
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        #     return loss, logits

        return logits