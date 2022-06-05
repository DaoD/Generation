import torch
import torch.nn as nn
import torch.nn.init as init


class GenOnlyModel(nn.Module):
    def __init__(self, bart, config):
        super(GenOnlyModel, self).__init__()
        self.bart_conditional = bart
        self.config = config
        # for classification
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.activation = nn.Tanh()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, batch_data, is_ret_test=False, is_encoding=False):
        if is_ret_test:
            batch_score = self.ret_test(batch_data)
            return batch_score
        elif is_encoding:
            context_rep = self.encode(batch_data)
            return context_rep
        else:
            context = batch_data["context_input_ids"]
            context_attn_mask = batch_data["context_attention_mask"]
            context_eos_pos = batch_data["context_eos_position"]
            response = batch_data["response_input_ids"]
            response_attn_mask = batch_data["response_attention_mask"]
            response_eos_pos = batch_data["response_eos_position"]
            response_lebels = batch_data["response_labels"]

            seq_outputs = self.bart_conditional(input_ids=context, attention_mask=context_attn_mask, labels=response_lebels)
            gen_loss = seq_outputs.loss
            ret_loss = torch.tensor([0.0]).to(torch.cuda.current_device())

            return gen_loss, ret_loss
