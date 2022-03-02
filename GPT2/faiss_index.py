import numpy as np
import faiss
import torch
from tqdm import tqdm
import random

np.random.seed(0)
def retrieval_test():
    d = 768
    query_vec = np.asarray(torch.load("./output/result/reddit3/test.10w.ctx.rep.pt"))
    doc_vec = np.asarray(torch.load("./output/result/reddit3/test.10w.rep.rep.pt"))
    ngpus = faiss.get_num_gpus()
    print(ngpus)
    cpu_index = faiss.IndexFlatIP(d)
    print(cpu_index.is_trained)
    # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index = cpu_index
    gpu_index.add(doc_vec)
    print(gpu_index.ntotal)

    all_ctx_vec = []
    all_ctx_index = []
    all_doc_index = []
    
    with open("./data/reddit3/test.10w.id.txt", "r") as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            line = line.strip().split("\t")
            ctx_idx = int(line[0])
            doc_idx = int(line[1])
            query_ctx = query_vec[ctx_idx]
            all_ctx_index.append(ctx_idx)
            all_ctx_vec.append(query_ctx)
            all_doc_index.append(doc_idx)

    all_ctx_vec = np.asarray(all_ctx_vec)
    k = 1000
    D, I = gpu_index.search(all_ctx_vec, k)
    all_results_dict = {}
    count_1000, count_100, count_10 = 0, 0 ,0
    for i in tqdm(range(I.shape[0])):
        index_list = list(I[i])
        doc_idx = all_doc_index[i]
        if doc_idx in index_list:
            count_1000 += 1
        if doc_idx in index_list[:100]:
            count_100 += 1
        if doc_idx in index_list[:10]:
            count_10 += 1

    print(count_1000 / I.shape[0], count_100 / I.shape[0], count_10 / I.shape[0])

retrieval_test()
