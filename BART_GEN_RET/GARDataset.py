import linecache
from torch.utils.data import Dataset
import numpy as np
    
class GARDataset(Dataset):
    def __init__(self, filename, tokenizer, dataset="personachat"):
        super(GARDataset, self).__init__()
        self._filename = filename
        assert dataset in ["personachat", "cmudog"]
        self._dataset = dataset
        if dataset == "personachat":
            self._max_ctx_len = 180
            self._max_rep_len = 50
            self._max_doc_len = 90
            self._max_doc_num = 1
        else:
            self._max_ctx_len = 256
            self._max_rep_len = 50
            self._max_doc_len = 200
            self._max_doc_num = 7
        self._max_seq_len = self._max_doc_len + self._max_ctx_len + self._max_rep_len + 3
        self._tokenizer = tokenizer
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        utterances = (line[0] + " ").split(' _eos_ ')[:-1]

        context_tokens, context_segment = [], []
        for utterance in utterances:
            u_tokens = self._tokenizer.tokenize(utterance) + ["[eot]"]
            context_tokens.extend(u_tokens)
        context_tokens = context_tokens[-self._max_ctx_len:]

        resp = line[1]
        r_tokens = self._tokenizer.tokenize(resp)
        r_tokens += ["[eot]"]
        r_segment = [1] * len(r_tokens)
        r_tokens = r_tokens[-self._max_rep_len:]

        if line[5] == "NA":
            has_next = 0
        else:
            has_next = 1
        next_utterance = self._tokenizer.tokenize(line[5])
        next_utterance = next_utterance[:(self._max_rep_len - 2)]
        
        label = float(line[2])

        if self._dataset == "personachat":
            if line[3] != "NA":
                documents1 = line[3].split("|")
                documents2 = line[4].split("|")
                documents = documents1 + documents2
            else:
                documents2 = line[4].split("|")
                documents = documents2
            document_tokens = []
            for sent in documents:
                s_tokens = self._tokenizer.tokenize(sent) + ["[eop]"]
                document_tokens.extend(s_tokens)
            document_tokens = document_tokens[:self._max_doc_len]
            seq_tokens = [self._tokenizer.bos_token] + document_tokens + context_tokens + [self._tokenizer.sep_token] + r_tokens + [self._tokenizer.sep_token]
            eos_position = len(seq_tokens) - 1
            attention_mask = [1] * len(seq_tokens)
            assert len(seq_tokens) == len(attention_mask) and len(seq_tokens) <= self._max_seq_len
            while len(seq_tokens) < self._max_seq_len:
                seq_tokens.append(self._tokenizer.pad_token)
                attention_mask.append(0)
            eos_mask = [0] * len(seq_tokens)
            eos_mask[eos_position] = 1
            seq_tokens = self._tokenizer.convert_tokens_to_ids(seq_tokens)
            next_utterance = [self._tokenizer.bos_token] + next_utterance + [self._tokenizer.eos_token]
            next_utterance_ids = self._tokenizer.convert_tokens_to_ids(next_utterance)
            assert len(next_utterance_ids) <= self._max_rep_len
            while len(next_utterance_ids) < self._max_rep_len:
                next_utterance_ids.append(-100)
            batch = {
                "sequence_input_ids": np.asarray(seq_tokens),
                "sequence_attention_mask": np.asarray(attention_mask),
                "eos_position": np.asarray(eos_mask),
                "next_utterance_ids": np.asarray(next_utterance_ids, dtype=np.int64),
                "labels": label,
                "has_next_label": has_next,
            }
        else:
            documents = line[3].split("|")
            documents_long = " ".join(documents)
            document_long_tokens = self._tokenizer.tokenize(documents_long)
            start = 0
            all_passage_tokens = []
            all_input_ids, all_segment_ids, all_attention_masks = [], [], []
            while start < len(document_long_tokens):
                passage_tokens = document_long_tokens[start: start + 200]
                passage_segment = [0] * len(passage_tokens)
                seq_tokens = ["[CLS]"] + passage_tokens + context_tokens + ["[SEP]"] + r_tokens + ["[SEP]"]
                seq_segment = [0] + passage_segment + context_segment + [0] + r_segment + [1]
                attention_mask = [1] * len(seq_tokens)
                assert len(seq_tokens) == len(seq_segment) == len(attention_mask) and len(seq_tokens) <= self._max_seq_len
                while len(seq_tokens) < self._max_seq_len:
                    seq_tokens.append("[PAD]")
                    seq_segment.append(0)
                    attention_mask.append(0)
                seq_tokens = self._tokenizer.convert_tokens_to_ids(seq_tokens)
                all_input_ids.append(seq_tokens)
                all_segment_ids.append(seq_segment)
                all_attention_masks.append(attention_mask)
                start += 190
            assert len(all_passage_tokens) <= self._max_doc_num
            while len(all_input_ids) < self._max_doc_num:
                seq_tokens = ["[CLS]"] + context_tokens + ["[SEP]"] + r_tokens + ["[SEP]"]
                seq_segment = [0] + context_segment + [0] + r_segment + [1]
                attention_mask = [1] * len(seq_tokens)
                assert len(seq_tokens) == len(seq_segment) == len(attention_mask) and len(seq_tokens) <= self._max_seq_len
                while len(seq_tokens) < self._max_seq_len:
                    seq_tokens.append("[PAD]")
                    seq_segment.append(0)
                    attention_mask.append(0)
                seq_tokens = self._tokenizer.convert_tokens_to_ids(seq_tokens)
                all_input_ids.append(seq_tokens)
                all_segment_ids.append(seq_segment)
                all_attention_masks.append(attention_mask)
            batch = {
                "input_ids": np.asarray(all_input_ids),
                "segment_ids": np.asarray(all_segment_ids),
                "attention_mask": np.asarray(all_attention_masks),
                "labels": label
            }

        return batch
    
    def __len__(self):
        return self._total_data

