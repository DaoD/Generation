import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
from PrefixEncoder import PrefixEncoder


class BertPrefixSS(nn.Module):
    def __init__(self, bert_model):
        super(BertPrefixSS, self).__init__()
        self.bert_model = bert_model
        self.pre_seq_len = 20
        self.n_layer = 24
        self.n_head = 16
        self.hidden = 1024
        self.n_embd = self.hidden // self.n_head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden, 1)
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(False, self.pre_seq_len, self.hidden, 512, self.n_layer)

        bert_param = 0
        for name, param in self.bert_model.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print("total param is: ", total_param)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert_model.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(batch_size, self.pre_seq_len, self.n_layer * 2, self.n_head, self.n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, batch_data):
        """
        Args:
            batch_data: a dict
        """
        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        token_type_ids = batch_data["token_type_ids"]
        # position_ids = batch_data["position_ids"]
        # inputs_embeds = batch_data["inputs_embeds"]

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert_model.device)
        attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        outputs = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                  past_key_values=past_key_values)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits.squeeze(1)


class BertPromptSS(nn.Module):
    def __init__(self, bert_model):
        super(BertPromptSS, self).__init__()
        self.bert_model = bert_model
        self.embeddings = self.bert_model.embeddings
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.pre_seq_len = 20
        self.n_layer = 24
        self.n_head = 16
        self.n_embd = 1024 // self.n_head

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = nn.Embedding(self.pre_seq_len, 768)

        bert_param = 0
        for name, param in self.bert_model.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print("total param is: ", total_param)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert_model.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(self, batch_data):
        """
        Args:
            batch_data: a dict
        """
        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        token_type_ids = batch_data["token_type_ids"]
        batch_size = input_ids.shape[0]

        raw_embedding = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        prompts = self.get_prompt(batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert_model.device)
        attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        outputs = self.bert_model(attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        first_token_tensor = sequence_output[:, 0]
        pooled_output = self.bert_model.pooler.dense(first_token_tensor)
        pooled_output = self.bert_model.pooler.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits.squeeze(1)