import torch
from tqdm import tqdm
import random

class Data:
    def __init__(self, filename, max_seq_length):
        self._max_seq_len = max_seq_length
        self._max_doc_len = 40
        self._all_context_ids = []
        self._all_docs_ids = []
        self._all_labels = []
        with open(filename, "r") as fr:
            lines = fr.readlines()
            for line in tqdm(lines, leave=False):
                line = line.strip().split("\t")
                context = line[0].split()
                docs = line[1].split("[=====]")
                labels = line[2].split("[===]")
                self._all_context_ids.append([int(x) for x in context])
                doc_ids = []
                for doc in docs:
                    doc_ids.append([int(x) for x in doc.split()])
                self._all_docs_ids.append(doc_ids)
                self._all_labels.append([int(x) for x in labels])
            tqdm.write("Load data finished...")
        self._sample_num = len(self._all_context_ids)
        self._all_train_ids = list(range(self._sample_num))
        
    def get_train_next_batch(self, batch_size, pacing_value):
        # train_ids = self._all_train_ids[:pacing_num]
        pacing_num = int(self._sample_num * pacing_value)
        train_ids = self._all_train_ids[:pacing_num]
        batch_idx_list = random.sample(train_ids, batch_size)
        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_masks = []
        batch_labels = []
        for idx in batch_idx_list:
            context_ids = self._all_context_ids[idx]
            docs_ids = self._all_docs_ids[idx]
            labels = self._all_labels[idx]
            docs_sample_ids = []
            docs_token_type_ids = []
            docs_attention_masks = []
            for doc_ids in docs_ids:
                doc_ids = doc_ids[:self._max_doc_len]
                sample_ids = context_ids + [102] + doc_ids + [102]
                token_type_ids = [0] * (len(context_ids) + 1) + [1] * (len(doc_ids) + 1)
                sample_ids = sample_ids[-(self._max_seq_len - 1):]
                token_type_ids = token_type_ids[-(self._max_seq_len - 1):]
                sample_ids = [101] + sample_ids
                token_type_ids = [0] + token_type_ids
                attention_mask = [1] * len(sample_ids)
                assert len(sample_ids) <= self._max_seq_len
                while len(sample_ids) < self._max_seq_len:
                    sample_ids.append(0)
                    token_type_ids.append(0)
                    attention_mask.append(0)
                assert len(sample_ids) == len(token_type_ids) == len(attention_mask) == self._max_seq_len
                docs_sample_ids.append(sample_ids)
                docs_token_type_ids.append(token_type_ids)
                docs_attention_masks.append(attention_mask)
            batch_input_ids.append(docs_sample_ids)
            batch_token_type_ids.append(docs_token_type_ids)
            batch_attention_masks.append(docs_attention_masks)
            batch_labels.append(labels)
        batch = {
            'input_ids': torch.LongTensor(batch_input_ids), 
            'token_type_ids': torch.LongTensor(batch_token_type_ids), 
            'attention_mask': torch.LongTensor(batch_attention_masks), 
            'labels': torch.FloatTensor(batch_labels)
        }
        return batch