import torch
import torch.nn as nn


class PrefixEncoder(nn.Module):
    def __init__(self, prefix_projection=True, pre_seq_len=10, hidden_size=768, prefix_hidden_size=512, num_hidden_layers=12):
        super().__init__()
        self.prefix_projection = prefix_projection
        self.pre_seq_len = pre_seq_len
        self.hidden_size = hidden_size
        self.prefix_hidden_size = prefix_hidden_size
        self.num_hidden_layers = num_hidden_layers
        if self.prefix_projection:
            self.embedding = nn.Embedding(self.pre_seq_len, self.hidden_size)
            self.trans = nn.Sequential(
                nn.Linear(self.hidden_size, self.prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(self.prefix_hidden_size, self.num_hidden_layers * 2 * self.hidden_size)
            )
        else:
            self.embedding = nn.Embedding(self.pre_seq_len, self.num_hidden_layers * 2 * self.hidden_size)

    def forward(self, prefix):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values