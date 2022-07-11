import linecache
from torch.utils.data import Dataset
import numpy as np

class FileEncodingDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len=82, dataset="reddit3"):
        super(FileEncodingDataset, self).__init__()
        self._filename = filename
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_response_len = max_len
        with open(filename, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines())
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")

        response = line[1]
        response_encode = self._tokenizer(response, padding="max_length", max_length=self._max_response_len, truncation=True)
        response_input_ids = response_encode.input_ids
        response_attention_mask = response_encode.attention_mask
        response_eos_position = response_input_ids.index(self._tokenizer.eos_token_id)
        response_eos_pos = [0] * self._max_response_len
        response_eos_pos[response_eos_position] = 1

        batch = {
            "context_input_ids": np.asarray(response_input_ids),
            "context_attention_mask": np.asarray(response_attention_mask),
            "context_eos_position": np.asarray(response_eos_pos),
        }
        return batch

    def __len__(self):
        return self._total_data

