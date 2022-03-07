import torch
import torch.nn as nn
import torch.nn.init as init

class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class BartGAR(nn.Module):
    def __init__(self, bart, config):
        super(BartGAR, self).__init__()
        self.bart_conditional = bart
        self.config = config
        # for classification
        self.pooler = BartClassificationHead(config.d_model, config.d_model, 1, 0.0)
    
    def forward(self, batch_data, is_test=False):
        sequence = batch_data["sequence_input_ids"]
        sequence_attn_mask = batch_data["sequence_attention_mask"]
        classification_head_token = batch_data["eos_position"]

        batch_size = sequence.size(0)
        sequence_outputs = self.bart_conditional.model(input_ids=sequence, attention_mask=sequence_attn_mask)
        sequence_hidden = sequence_outputs[0]
        classification_head_token = (classification_head_token == 1)
        sequence_score = self.pooler(sequence_hidden[classification_head_token, :]).squeeze(1)

        if is_test:
            return sequence_score
        else:
            next_utterance = batch_data["next_utterance_ids"]
            has_next_label = batch_data["has_next_label"]
            gen_loss = torch.tensor(0.0).to(torch.cuda.current_device())
            for idx, label in enumerate(has_next_label):
                if label == 1:
                    seq_outputs = self.bart_conditional(input_ids=sequence[idx].unsqueeze(0), attention_mask=sequence_attn_mask[idx].unsqueeze(0), labels=next_utterance[idx].unsqueeze(0))
                    gen_loss += seq_outputs.loss
            has_labels = torch.sum(has_next_label)
            if has_labels > 0:
                gen_loss /= has_labels
            return sequence_score, gen_loss
