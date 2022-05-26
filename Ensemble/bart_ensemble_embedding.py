import torch
import torch.nn as nn


class BartEnsembleEmbeddingModel(nn.Module):
    def __init__(self, bart, config):
        super(BartEnsembleEmbeddingModel, self).__init__()
        self.bart_conditional = bart
        self.config = config
        self.loss_weights = nn.Parameter(torch.ones(2))
        # for classification
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.activation = nn.Tanh()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, batch_data, is_test=False, is_encoding=False):
        context = batch_data["context_input_ids"]
        context_attn_mask = batch_data["context_attention_mask"]
        context_eos_pos = batch_data["context_eos_position"]

        response = batch_data["response_input_ids"]
        response_attn_mask = batch_data["response_attention_mask"]
        response_eos_pos = batch_data["response_eos_position"]
        response_labels = batch_data["response_labels"]

        batch_size = context.size(0)

        context_outputs = self.bart_conditional.model(input_ids=context, attention_mask=context_attn_mask)
        context_hidden = context_outputs.encoder_last_hidden_state 
        context_cls_pos = (context_eos_pos == 1)
        context_rep = self.activation(self.dense(self.dropout(context_hidden[context_cls_pos, :])))

        embedding_matrix = self.bart_conditional.get_input_embeddings().weight  # vocab_size, hidden_size
        # batch_sim =  torch.einsum("ad,bd->ab", context_rep, response_rep)
        # batch_ret_label = torch.arange(batch_size).to(torch.cuda.current_device())
        # ret_loss = self.ce_loss(batch_sim, batch_ret_label)
        seq_outputs = self.bart_conditional(input_ids=context, attention_mask=context_attn_mask, labels=response_labels)
        logits = seq_outputs.logits  # batch, seq_len, vocab_size
        predict_embedding = torch.einsum("blv,vd->bld", logits, embedding_matrix)  # batch, seq_len, hidden
        resposne_outputs = self.bart_conditional.model(inputs_embeds=predict_embedding, attention_mask=response_attn_mask, decoder_inputs_embeds=predict_embedding)

        # resposne_outputs = self.bart_conditional.model(input_ids=response, attention_mask=response_attn_mask)
        response_hidden = resposne_outputs.encoder_last_hidden_state
        response_cls_pos = (response_eos_pos == 1)
        response_rep = self.activation(self.dense(self.dropout(response_hidden[response_cls_pos, :])))
        batch_sim =  torch.einsum("ad,bd->ab", context_rep, response_rep)
        batch_ret_label = torch.arange(batch_size).to(torch.cuda.current_device())
        ret_loss = self.ce_loss(batch_sim, batch_ret_label)

        gen_loss = seq_outputs.loss

        loss = (ret_loss / (self.loss_weights[0] * 2)) + (self.loss_weights[0] + 1).log() + (gen_loss / (self.loss_weights[1] * 2)) + (self.loss_weights[1] + 1).log()

        return gen_loss, ret_loss, loss
