import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# BART only for retrieval, using hard negative + in batch negative

class RetOnlyModel(nn.Module):
    def __init__(self, bart, config):
        super(RetOnlyModel, self).__init__()
        self.bart_conditional = bart
        self.config = config
        # for classification
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.activation = nn.Tanh()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, batch_data, is_test=False, is_encoding=False):
        """
            context_input_ids: [batch, seq_len]
            context_attention_mask: [batch, seq_len]
            context_eos_position: [batch, seq_len]
            candidate response includes one positive response and n negative responses
            cand_response_input_ids: [batch, cand_num, seq_len]
            cand_response_attention_mask: [batch, cand_num, seq_len]
            cand_response_eos_position: [batch, cand_num, seq_len]
        """
        context = batch_data["context_input_ids"]
        context_attn_mask = batch_data["context_attention_mask"]
        context_eos_pos = batch_data["context_eos_position"]

        context_outputs = self.bart_conditional.model(input_ids=context, attention_mask=context_attn_mask)
        context_hidden = context_outputs.encoder_last_hidden_state 
        context_cls_pos = (context_eos_pos == 1)
        context_rep = self.activation(self.dense(self.dropout(context_hidden[context_cls_pos, :])))

        batch_size = context.size(0)

        if is_encoding:
            return context_rep

        response = batch_data["response_input_ids"]
        response_attn_mask = batch_data["response_attention_mask"]
        response_eos_pos = batch_data["response_eos_position"]
        resposne_outputs = self.bart_conditional.model(input_ids=response, attention_mask=response_attn_mask)
        response_hidden = resposne_outputs.encoder_last_hidden_state 
        response_cls_pos = (response_eos_pos == 1)
        response_rep = self.activation(self.dense(self.dropout(response_hidden[response_cls_pos, :])))

        batch_score = torch.einsum("bd,cd->bc", context_rep, response_rep)
        cand_response = batch_data["cand_response_input_ids"]
        cand_response_attn_mask = batch_data["cand_response_attention_mask"]
        cand_response_eos_pos = batch_data["cand_response_eos_position"]
        r1, r2, r3 = cand_response.size()
        cand_response_flat = cand_response.reshape(r1 * r2, r3)
        cand_response_attn_flat = cand_response_attn_mask.reshape(r1 * r2, r3)
        cand_response_eos_pos_flat = cand_response_eos_pos.reshape(r1 * r2, r3)
        cand_resposne_outputs = self.bart_conditional.model(input_ids=cand_response_flat, attention_mask=cand_response_attn_flat)
        cand_response_hidden = cand_resposne_outputs.encoder_last_hidden_state
        cand_response_cls_pos_flat = (cand_response_eos_pos_flat == 1)
        cand_response_rep_flat = self.activation(self.dense(self.dropout(cand_response_hidden[cand_response_cls_pos_flat, :])))  # [batch * cand_num, hidden]
        cand_score = torch.einsum("bd,cd->bc", context_rep, cand_response_rep_flat)  # [batch, batch * cand_num]
        all_scores = torch.cat([batch_score, cand_score], dim=1)  # [batch, batch + batch * cand_num]

        if is_test:
            return all_matching_score
        else:
            batch_ret_label = torch.arange(batch_size).to(torch.cuda.current_device())
            ret_loss = self.ce_loss(all_scores, batch_ret_label)  # error!
            return ret_loss
