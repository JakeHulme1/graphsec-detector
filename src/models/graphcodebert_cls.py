from transformers import AutoConfig, AutoModelForSequenceClassification

class GCBertClassifier:
    """
    Wrapper around HuggingFace's GraphCodeBERT model for sequence classification.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: configuration object or dict with arributes/keys:
                - model_name_or_path (str): pretrained model identifier
                - num_labels (int): number of output classes
                - hidden_dropout_prob (float): dropout probability for hidden layers
                - classifier_dropout (float): dropout for the classifier head
        """
        #  Load and adjust configuration
        config = AutoConfig.from_pretrained(
            cfg.model_name_or_path,
            num_labels=cfg.num_labels,
            hidden_dropout_prob=getattr(cfg, 'hidden_dropout_prob', None),
            classifier_dropout=getattr(cfg, 'classifier_dropout', None),
        )

        # INstantiate the model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name_or_path,
            config=config)
        
    def forward(self, input_ids, attention_mask, node_type_ids=None, node_mask=None, edge_index=None, labels=None):
        """
        Forward pass for classification

        Args:
            - input_ids (LongTensor): token IDs [batch_size, seq_len]
            - attention_mask (torch.LongTensor): attention mask, shape (batch_size, seq_len)
            - node_type_ids (torch.LongTensor, optional): node type IDs, shape (batch_size, max_nodes)
            - node_mask (torch.LongTensor, optional): node mask, shape (batch_size, max_nodes)
            - edge_index (torch.LongTensor, optional): edge indices, shape (2, num_edges)
            - labels (torch.LongTensor, optional): classification labels, shape (batch_size,)
        
        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: containing loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            node_type_ids=node_type_ids,
            node_mask=node_mask,
            edge_index=edge_index,
            labels=labels,
        )
        return outputs