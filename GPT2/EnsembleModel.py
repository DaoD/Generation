import torch
import torch.nn as nn
import torch.nn.init as init

class EnsembleModel(nn.Module):
    def __init__(self, dialogpt):
        super(EnsembleModel, self).__init__()
        self.model = dialogpt
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, batch_data, is_ret_test=False, is_encoding=False):
        """
        Args:

        """
        if is_ret_test:
            batch_score = self.ret_test(batch_data)
            return batch_score
        elif is_encoding:
            context_rep = self.encode(batch_data)
            return context_rep
        else:
            samples = batch_data["samples"]
            labels = batch_data["labels"]
            context = batch_data["context"]
            context_label = batch_data["context_label"]
            reply = batch_data["reply"]
            reply_label = batch_data["reply_label"]

            batch_size = samples.size(0)

            context_outputs = self.model(input_ids=context, output_hidden_states=True)
            context_hidden = context_outputs.hidden_states
            context_rep = self.activation(self.dense(context_hidden[-1])) * context_label.unsqueeze(-1)
            context_rep = torch.sum(context_rep, dim=1)  # [batch, hidden]

            resp_outputs = self.model(input_ids=reply, output_hidden_states=True)
            resp_hidden = resp_outputs.hidden_states
            resp_rep = self.activation(self.dense(resp_hidden[-1])) * reply_label.unsqueeze(-1)
            resp_rep = torch.sum(resp_rep, dim=1)  # [batch, hidden]

            outputs = self.model(input_ids=samples, labels=labels)
            gen_loss = outputs.loss
            batch_sim =  torch.einsum("ad,bd->ab", context_rep, resp_rep)
            batch_ret_label = torch.arange(batch_size).to(torch.cuda.current_device())
            ret_loss = self.ce_loss(batch_sim, batch_ret_label)
            return gen_loss, ret_loss

    def ret_test(self, batch_data):
        samples = batch_data["samples"]
        labels = batch_data["labels"]
        context = batch_data["context"]
        context_label = batch_data["context_label"]
        reply = batch_data["reply"]
        reply_label = batch_data["reply_label"]

        batch_size = samples.size(0)

        context_outputs = self.model(input_ids=context, output_hidden_states=True)
        context_hidden = context_outputs.hidden_states
        context_rep = self.activation(self.dense(context_hidden[-1])) * context_label.unsqueeze(-1)
        context_rep = torch.sum(context_rep, dim=1)  # [batch, hidden]

        resp_outputs = self.model(input_ids=reply, output_hidden_states=True)
        resp_hidden = resp_outputs.hidden_states
        resp_rep = self.activation(self.dense(resp_hidden[-1])) * reply_label.unsqueeze(-1)
        resp_rep = torch.sum(resp_rep, dim=1)  # [batch, hidden]

        batch_score = torch.einsum("bd,bd->b", context_rep, resp_rep)
        return batch_score

    def encode(self, batch_data):
        context = batch_data["sent"]
        context_label = batch_data["sent_label"]

        context_outputs = self.model(input_ids=context, output_hidden_states=True)
        context_hidden = context_outputs.hidden_states
        context_rep = self.activation(self.dense(context_hidden[-1])) * context_label.unsqueeze(-1)
        context_rep = torch.sum(context_rep, dim=1)  # [batch, hidden]

        return context_rep