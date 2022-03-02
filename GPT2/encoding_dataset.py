import linecache
from torch.utils.data import Dataset
import numpy as np

class FileEncodingDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len=81, dataset="reddit3"):
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
        response_id = self._tokenizer.encode(response)
        response_id = response_id[:self._max_response_len - 1]

        reply = response_id + [self._tokenizer.eos_token_id]
        reply_label_pos = len(response_id)
        while len(reply) < self._max_response_len:
            reply.append(0)
        reply_label = [0] * len(reply)
        reply_label[reply_label_pos] = 1

        batch = {
            "sent": np.asarray(reply),
            "sent_label": np.asarray(reply_label)
        }
        return batch

    def __len__(self):
        return self._total_data

