import torch
import torch.nn as nn
import torch.nn.init as init


# class BartClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""
#     def __init__(
#         self,
#         input_dim: int,
#         inner_dim: int,
#         num_classes: int,
#         pooler_dropout: float,
#     ):
#         super().__init__()
#         self.dense = nn.Linear(input_dim, inner_dim)
#         self.dropout = nn.Dropout(p=pooler_dropout)
#         self.out_proj = nn.Linear(inner_dim, num_classes)

#     def forward(self, hidden_states: torch.Tensor):
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.dense(hidden_states)
#         hidden_states = torch.tanh(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.out_proj(hidden_states)
#         return hidden_states

class EnsembleModel(nn.Module):
    def __init__(self, bart, config):
        super(EnsembleModel, self).__init__()
        self.bart_conditional = bart
        self.config = config
        # for classification
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.classifier_dropout)
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
            context = batch_data["context_input_ids"]
            context_attn_mask = batch_data["context_attention_mask"]
            response = batch_data["response_input_ids"]
            resposne_attn_mask = batch_data["resposne_attention_mask"]
            classification_head_token = batch_data["head_token"]
            response_lebels = batch_data["response_labels"]

            batch_size = context.size(0)

            context_outputs = self.bart_conditional.model(input_ids=context, attention_mask=context_attn_mask, decoder_input_ids=classification_head_token)
            context_hidden = context_outputs.encoder_last_hidden_state 
            context_rep = self.activation(self.dense(self.dropout(context_hidden[:, 0, :])))

            resposne_outputs = self.bart_conditional.model(input_ids=response, attention_mask=resposne_attn_mask, decoder_input_ids=classification_head_token)
            response_hidden = resposne_outputs.encoder_last_hidden_state
            response_rep = self.activation(self.dense(self.dropout(response_hidden[:, 0, :])))

            batch_sim =  torch.einsum("ad,bd->ab", context_rep, response_rep)
            batch_ret_label = torch.arange(batch_size).to(torch.cuda.current_device())
            ret_loss = self.ce_loss(batch_sim, batch_ret_label)

            seq_outputs = self.bart_conditional(input_ids=context, attention_mask=context_attn_mask, labels=response_lebels)
            gen_loss = seq_outputs.loss

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
        context = batch_data["input_ids"]
        context_attn_mask = batch_data["attention_mask"]
        classification_head_token = batch_data["head_token"]

        context_outputs = self.model(input_ids=context, attention_mask=context_attn_mask, decoder_input_ids=classification_head_token)
        context_hidden = context_outputs.encoder_last_hidden_state 
        context_rep = self.activation(self.dense(self.dropout(context_hidden[:, 0, :])))

        return context_rep