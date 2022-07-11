import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Ensemble model, using hard negative for retrieval

class EnsembleModel(nn.Module):
    def __init__(self, bart, config):
        super(EnsembleModel, self).__init__()
        self.bart_conditional = bart
        self.config = config
        self.loss_weights = nn.Parameter(torch.ones(2))
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
        single_positive_score = torch.diagonal(batch_score, 0)
        positive_scores = single_positive_score.reshape(-1, 1).repeat(1, batch_size).reshape(-1)
        rel_pair_mask = 1 - torch.eye(batch_size, dtype=batch_score.dtype, device=batch_score.device)
        batch_score = batch_score.reshape(-1)
        logit_matrix = torch.cat([positive_scores.unsqueeze(1), batch_score.unsqueeze(1)], dim=1)
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0] * rel_pair_mask.reshape(-1)
        first_loss, first_num = loss.sum(), rel_pair_mask.sum()

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
        cand_scores = cand_score.reshape(-1)
        positive_scores = single_positive_score.reshape(-1, 1).repeat(1, r1 * r2).reshape(-1)
        cand_logit_matrix = torch.cat([positive_scores.unsqueeze(1), cand_scores.unsqueeze(1)], dim=1)
        cand_lsm = F.log_softmax(cand_logit_matrix, dim=1)
        cand_loss = -1.0 * cand_lsm[:, 0]
        second_loss, second_num = cand_loss.sum(), len(cand_loss)

        if is_test:
            return all_matching_score
        else:
            response_labels = batch_data["response_labels"]
            ret_loss = (first_loss + second_loss) / (first_num + second_num)
            seq_outputs = self.bart_conditional(input_ids=context, attention_mask=context_attn_mask, labels=response_labels)
            gen_loss = seq_outputs.loss
            loss = (ret_loss / (self.loss_weights[0] * 2)) + (self.loss_weights[0] + 1).log() + (gen_loss / (self.loss_weights[1] * 2)) + (self.loss_weights[1] + 1).log()
            return gen_loss, ret_loss, loss
