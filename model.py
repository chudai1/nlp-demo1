import torch.nn as nn
from transformers import BertModel
import config

class TextClassifier(nn.Module):
    def __init__(self, model_name=config.model_name, num_labels=None):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name) 
        self.dropout = nn.Dropout(config.dropout) 
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}
