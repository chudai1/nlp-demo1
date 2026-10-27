import torch.nn as nn
from transformers import BertModel
import config

class TextClassifier(nn.Module):
    def __init__(self, model_name=config.model_name, num_labels=config.num_labels):
        super(TextClassifier, self).__init__()
        # 主干
        self.bert = BertModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 取 [CLS] 向量
        pooled_output = outputs.pooler_output
        
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return {"loss": loss, "logits": logits}