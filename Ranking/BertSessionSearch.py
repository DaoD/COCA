import torch.nn as nn
import torch.nn.init as init

class BertSessionSearch(nn.Module):
    def __init__(self, bert_model):
        super(BertSessionSearch, self).__init__()
        self.bert_model = bert_model
        self.classifier = nn.Linear(768, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        init.xavier_normal_(self.classifier.weight)
    
    def forward(self, batch_data):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """
        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        token_type_ids = batch_data["token_type_ids"]
        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        sent_rep = self.dropout(self.bert_model(**bert_inputs)[1])
        y_pred = self.classifier(sent_rep)
        return y_pred.squeeze(1)
