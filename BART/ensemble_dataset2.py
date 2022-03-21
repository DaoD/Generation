import linecache
from torch.utils.data import Dataset
import numpy as np

class FileDataset(Dataset):
    def __init__(self, filename, tokenizer, dataset="reddit3"):
        super(FileDataset, self).__init__()
        self._filename = filename
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_context_len = 82
        self._max_response_len = 42
        self._max_seq_len = self._max_context_len + self._max_response_len
        with open(filename, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines())
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")

        context = line[0]
        pos_response = line[1]
        neg_responses = line[2:]

        context_encode = self._tokenizer(context, padding="max_length", max_length=self._max_context_len, truncation=True)
        context_context_input_ids = context_encode.input_ids
        context_attention_mask = context_encode.attention_mask
        context_eos_position = context_context_input_ids.index(self._tokenizer.eos_token_id)
        context_eos_pos = [0] * self._max_context_len
        context_eos_pos[context_eos_position] = 1

        response_encode = self._tokenizer(pos_response, padding="max_length", max_length=self._max_response_len, truncation=True)
        response_input_ids = response_encode.input_ids
        response_attention_mask = response_encode.attention_mask
        response_eos_position = response_input_ids.index(self._tokenizer.eos_token_id)
        response_eos_pos = [0] * self._max_response_len
        response_eos_pos[response_eos_position] = 1

        response_labels = np.asarray(response_input_ids)
        response_labels[response_labels == self._tokenizer.pad_token_id] = -100

        group_neg_response_ids = []
        group_neg_response_attentions = []
        group_neg_response_eos_positions = []
        for neg_response in neg_responses:
            neg_response_encode = self._tokenizer(neg_response, padding="max_length", max_length=self._max_response_len, truncation=True)
            neg_response_input_ids = neg_response_encode.input_ids
            neg_response_attention_mask = neg_response_encode.attention_mask
            neg_response_eos_position = neg_response_input_ids.index(self._tokenizer.eos_token_id)
            neg_response_eos_pos = [0] * self._max_response_len
            neg_response_eos_pos[response_eos_position] = 1
            group_neg_response_ids.append(neg_response_input_ids)
            group_neg_response_attentions.append(neg_response_attention_mask)
            group_neg_response_eos_positions.append(neg_response_eos_pos)

        batch = {
            "context_input_ids": np.asarray(context_context_input_ids),
            "context_attention_mask": np.asarray(context_attention_mask),
            "context_eos_position": np.asarray(context_eos_pos),
            "response_input_ids": np.asarray(response_input_ids),
            "response_attention_mask": np.asarray(response_attention_mask),
            "response_eos_position": np.asarray(response_eos_pos),
            "cand_response_input_ids": np.asarray(group_neg_response_ids),
            "cand_response_attention_mask": np.asarray(group_neg_response_attentions),
            "cand_response_eos_position": np.asarray(group_neg_response_eos_positions),
            "response_labels": np.asarray(response_labels, dtype=np.int64)
        }

        return batch

    def __len__(self):
        return self._total_data

