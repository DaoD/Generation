import linecache
from torch.utils.data import Dataset
import numpy as np

class FileDataset(Dataset):
    def __init__(self, filename, tokenizer, dataset="reddit3"):
        super(FileDataset, self).__init__()
        self._filename = filename
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_context_len = 81
        self._max_response_len = 41
        self._max_seq_len = self._max_context_len + self._max_response_len
        with open(filename, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines())
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")

        utterance_ids = []
        for utterance in line[:-1]:
            utterance_ids.extend(self._tokenizer.encode(utterance))
        utterance_ids = utterance_ids[-(self._max_context_len - 1):]
        response = line[-1]
        response_id = self._tokenizer.encode(response)
        response_id = response_id[:self._max_response_len - 1]

        context = utterance_ids + [self._tokenizer.eos_token_id]
        context_label_pos = len(utterance_ids)
        while len(context) < self._max_context_len:
            context.append(0)
        context_label = [0] * len(context)
        context_label[context_label_pos] = 1

        reply = response_id + [self._tokenizer.eos_token_id]
        reply_label_pos = len(response_id)
        while len(reply) < self._max_response_len:
            reply.append(0)
        reply_label = [0] * len(reply)
        reply_label[reply_label_pos] = 1

        sample = utterance_ids + [self._tokenizer.eos_token_id] + response_id + [self._tokenizer.eos_token_id]
        labels = [-100] * len(utterance_ids) + [-100] + response_id + [self._tokenizer.eos_token_id]
        while len(sample) < self._max_seq_len:
            sample.append(0)
            labels.append(-100)
        batch = {
            "samples": np.asarray(sample, dtype=np.int64),
            "labels": np.asarray(labels, dtype=np.int64),
            "context": np.asarray(context),
            "context_label": np.asarray(context_label),
            "reply": np.asarray(reply),
            "reply_label": np.asarray(reply_label)
        }
        return batch

    def __len__(self):
        return self._total_data

