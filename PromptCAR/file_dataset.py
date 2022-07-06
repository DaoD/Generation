import linecache
from torch.utils.data import Dataset
import numpy as np


class FileDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer):
        super(FileDataset, self).__init__()
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
    
    def anno_main(self, qd_pairs):
        all_qd = []
        for qd in qd_pairs:
            qd = self._tokenizer.tokenize(qd)[:20]
            all_qd.append(qd)
        history = all_qd[:-2]
        query_tok = all_qd[-2]
        doc_tok = all_qd[-1]
        # history_toks = ["[CLS]"]
        history_toks = []
        segment_ids = []
        for iidx, sent in enumerate(history):
            history_toks.extend(sent + ["[eos]"])
            segment_ids.extend([0] * (len(sent) + 1))
        query_tok += ["[eos]"]
        query_tok += ["[SEP]"]
        doc_tok += ["[eos]"]
        doc_tok += ["[SEP]"]
        all_qd_toks = history_toks + query_tok + doc_tok
        segment_ids.extend([0] * len(query_tok))
        segment_ids.extend([1] * len(doc_tok))
        all_attention_mask = [1] * len(all_qd_toks)
        all_qd_toks = all_qd_toks[-(self._max_seq_length - 1):]
        segment_ids = segment_ids[-(self._max_seq_length - 1):]
        all_attention_mask = all_attention_mask[-(self._max_seq_length - 1):]
        all_qd_toks = ["[CLS]"] + all_qd_toks
        segment_ids = [0] + segment_ids
        all_attention_mask = [1] + all_attention_mask
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("[PAD]")
            segment_ids.append(0)
            all_attention_mask.append(0)
        assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
        input_ids = np.asarray(anno_seq)
        all_attention_mask = np.asarray(all_attention_mask)
        segment_ids = np.asarray(segment_ids)
        return input_ids, all_attention_mask, segment_ids
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        label = int(line[0])
        qd_pairs = line[1:]
        input_ids, attention_mask, segment_ids = self.anno_main(qd_pairs)
        batch = {
            'input_ids': input_ids, 
            'token_type_ids': segment_ids, 
            'attention_mask': attention_mask, 
            'labels': float(label)
        }
        return batch
    
    def __len__(self):
        return self._total_data

