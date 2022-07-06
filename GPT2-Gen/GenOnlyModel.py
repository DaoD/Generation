import torch
import torch.nn as nn
import torch.nn.init as init

class GenOnlyModel(nn.Module):
    def __init__(self, dialogpt):
        super(GenOnlyModel, self).__init__()
        self.model = dialogpt
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, batch_data, is_test=False):
        samples = batch_data["samples"]
        labels = batch_data["labels"]
        context = batch_data["context"]
        context_label = batch_data["context_label"]
        reply = batch_data["reply"]
        reply_label = batch_data["reply_label"]

        batch_size = samples.size(0)

        outputs = self.model(input_ids=samples, labels=labels)
        gen_loss = outputs.loss
        
        return gen_loss