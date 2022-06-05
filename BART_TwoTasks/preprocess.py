import random
import numpy as np
import torch
from tqdm import tqdm

def making_ret_set():
    response_dict = []
    with open("../data/reddit3/train.rep.dict.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            response_dict.append(line[1])
    with open("../data/reddit3/train.20w.txt", "r", encoding="utf-8") as fr:
        with open("../data/reddit3/train.1neg.txt", "w", encoding="utf-8") as fw:
            for line in fr:
                line = line.strip().split("\t")
                context = " ".join(line[:-1])
                response = line[-1]
                neg_response = random.choice(response_dict)
                while neg_response == response:
                    neg_response = random.choice(response_dict)
                fw.write(context + "\t" + response + "\t" + neg_response + "\n")
    
def making_ret_set2():
    response_dict = []
    with open("../output/result/reddit3/train.20w.prediction.bart.greedy.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            response_dict.append(line[0])
    with open("../data/reddit3/train.1neg.txt", "r", encoding="utf-8") as fr:
        with open("../data/reddit3/train.2neg.txt", "w", encoding="utf-8") as fw:
            for idx, line in enumerate(fr):
                line = line.strip().split("\t")
                context = line[0]
                pos_response = line[1]
                neg_response = line[2]
                gen_response = response_dict[idx]
                fw.write(context + "\t" + pos_response + "\t" + gen_response + "\t" + neg_response + "\n")

def test_generation_recall():
    with open("../output/result/reddit3/pred.result.21.dialogpt.greedy.txt", "r", encoding="utf-8") as fr:
        count = 0
        result_list = []
        r1, r2, r5, r10 = 0, 0, 0, 0
        all_sess_num = 0
        for line in fr:
            line = line.strip().split("\t")
            pred, label = float(line[0]), float(line[1])
            result_list.append((pred, label))
            count += 1
            if count % 21 == 0 and count != 0:
                np.random.shuffle(result_list)
                sort_data = sorted(result_list, key=lambda x: x[0], reverse=True)
                sort_label = [s_d[1] for s_d in sort_data]
                select_label = sort_label[:1]
                if 2 in select_label:
                    r1 += 1
                select_label = sort_label[:2]
                if 2 in select_label:
                    r2 += 1
                select_label = sort_label[:5]
                if 2 in select_label:
                    r5 += 1
                select_label = sort_label[:10]
                if 2 in select_label:
                    r10 += 1
                all_sess_num += 1
                result_list = []
        print(r1 / all_sess_num, r2 / all_sess_num, r5 / all_sess_num, r10 / all_sess_num)

def insert_generation_into_result():
    with open("../output/result/reddit3/test.result.bartgenonly.greedy.txt", "r", encoding="utf-8") as fr:
        with open("../data/reddit3/test.retrieval.20.txt", "r", encoding="utf-8") as fr2:
            with open("../data/reddit3/test.retrieval.21.bartgenonly.greedy.txt", "w", encoding="utf-8") as fw:
                preds = fr.readlines()
                lines = fr2.readlines()
                count = 0
                for idx, line in enumerate(lines):
                    line = line.strip().split("\t")
                    fw.write(line[0] + "\t" + line[1] + "\t" + line[2] + "\n")
                    if line[0] == "1":
                        pred_rep = preds[count].strip().split("\t")[0]
                        fw.write("2" + "\t" + line[1] + "\t" + pred_rep + "\n")
                        count += 1

def making_generated_golden_contrast_dataset():
    goldens = []
    with open("../data/reddit3/train.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            golden = line[-1]
            goldens.append(golden)
    with open("../output/result/reddit3/train.20w.prediction.bart.greedy.txt", "r", encoding="utf-8") as fr:
        with open("../data/reddit3/train.20w.contrast.txt", "w", encoding="utf-8") as fw:
            for idx, line in enumerate(fr):
                line = line.strip()
                fw.write(line + "\t" + goldens[idx] + "\n")

def get_faiss_retrieval_test_set():
    # with open("./data/reddit3/test.txt", "r", encoding="utf-8") as fr:
    #     with open("./data/reddit3/test.10w.txt", "w", encoding="utf-8") as fw:
    #         for idx, line in enumerate(fr):
    #             line = line.strip().split("\t")
    #             context = "\t".join(line[:-1])
    #             response = line[-1]
    #             fw.write(context + "\t" + response + "\n")
    #             if idx == 100000 - 1:
    #                 break
    ctx_dict = {}
    rep_dict = {}
    with open("../data/reddit3/test.5w.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            ctx = " ".join(line[:-1])
            if ctx not in ctx_dict:
                ctx_dict[ctx] = len(ctx_dict)
            if line[-1] not in rep_dict:
                rep_dict[line[-1]] = len(rep_dict)
    with open("../output/result/reddit3/test.result.bart.greedy.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            if line[0] not in rep_dict:
                rep_dict[line[0]] = len(rep_dict)
    ctx_dict_sort = sorted(ctx_dict.items(), key=lambda x: x[1])
    rep_dict_sort = sorted(rep_dict.items(), key=lambda x: x[1])
    with open("../data/reddit3/test.5w.ctx.dict.txt", "w", encoding="utf-8") as fw:
        for (ctx, idx) in ctx_dict_sort:
            fw.write(str(idx) + "\t" + ctx + "\n")
    with open("../data/reddit3/test.5w.rep.dict.txt", "w", encoding="utf-8") as fw:
        for (rep, idx) in rep_dict_sort:
            fw.write(str(idx) + "\t" + rep + "\n")

def transfer_sample_to_id():
    ctx_dict = {}
    rep_dict = {}
    with open("../data/reddit3/test.5w.ctx.dict.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            idx = line[0]
            ctx = line[1]
            ctx_dict[ctx] = idx
    with open("../data/reddit3/test.5w.rep.dict.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            idx = line[0]
            rep = line[1]
            rep_dict[rep] = idx
    gid = []
    with open("../output/result/reddit3/test.result.bart.greedy.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            gid.append(rep_dict[line[0]])
    with open("../data/reddit3/test.5w.txt", "r", encoding="utf-8") as fr:
        with open("../data/reddit3/test.5w.id.txt", "w", encoding="utf-8") as fw:
            for idx, line in enumerate(fr):
                line = line.strip().split("\t")
                ctx = " ".join(line[:-1])
                ctx_id = ctx_dict[ctx]
                rep_id = rep_dict[line[-1]]
                fw.write(ctx_id + "\t" + rep_id + "\t" + gid[idx] + "\n")
    
def make_ft_dataset():
    # response_dict = []
    # with open("../output/result/reddit3/train.20w.prediction.bart.greedy.txt", "r", encoding="utf-8") as fr:
    #     for line in fr:
    #         line = line.strip().split("\t")
    #         response_dict.append(line[0])
    # with open("../data/reddit3/train.1neg.txt", "r", encoding="utf-8") as fr:
    #     with open("../data/reddit3/train.ft.txt", "w", encoding="utf-8") as fw:
    #         for idx, line in enumerate(fr):
    #             line = line.strip().split("\t")
    #             context = line[0]
    #             pos_response = line[1]
    #             gen_response = response_dict[idx]
    #             fw.write(context + "\t" + pos_response + "\t" + gen_response + "\n")
    response_dict = []
    with open("../output/result/reddit3/test.result.bart.greedy.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            response_dict.append(line[0])
    with open("../data/reddit3/test.5w.txt", "r", encoding="utf-8") as fr:
        with open("../data/reddit3/test.ft.txt", "w", encoding="utf-8") as fw:
            for idx, line in enumerate(fr):
                line = line.strip().split("\t")
                context = " ".join(line[:-1])
                pos_response = line[-1]
                gen_response = response_dict[idx]
                fw.write(context + "\t" + pos_response + "\t" + gen_response + "\n")

# def check_result():
#     query_vec = np.asarray(torch.load("../output/result/reddit3/test.5w.bart.ft.ctx.pt"))
#     doc_vec = np.asarray(torch.load("../output/result/reddit3/test.5w.bart.ft.rep.pt"))
#     count = 0
#     with open("../data/reddit3/test.5w.id.txt", "r") as fr:
#         lines = fr.readlines()
#         for line in tqdm(lines):
#             line = line.strip().split("\t")
#             ctx_idx = int(line[0])
#             doc_idx = int(line[1])
#             gen_idx = int(line[2])
#             query_ctx = query_vec[ctx_idx]
#             doc = doc_vec[doc_idx]
#             gen = doc_vec[gen_idx]
#             if np.sum(query_ctx * doc) > np.sum(query_ctx * gen):
#                 count += 1
#     print(count / len(lines))

def make_concat_dataset():
    response_dict = []
    with open("../output/result/reddit3/train.20w.prediction.bart.greedy.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            response_dict.append(line[0])
    with open("../data/reddit3/train.1neg.txt", "r", encoding="utf-8") as fr:
        with open("../data/reddit3/train.concat.txt", "w", encoding="utf-8") as fw:
            for idx, line in enumerate(fr):
                line = line.strip().split("\t")
                context = line[0]
                pos_response = line[1]
                gen_response = response_dict[idx]
                context = context + " " + gen_response
                fw.write(context + "\t" + pos_response + "\n")
    response_dict = []
    with open("../output/result/reddit3/test.result.bart.greedy.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            response_dict.append(line[0])
    with open("../data/reddit3/test.5w.txt", "r", encoding="utf-8") as fr:
        with open("../data/reddit3/test.concat.txt", "w", encoding="utf-8") as fw:
            for idx, line in enumerate(fr):
                line = line.strip().split("\t")
                context = " ".join(line[:-1])
                pos_response = line[-1]
                gen_response = response_dict[idx]
                context = context + " " + gen_response
                fw.write(context + "\t" + pos_response + "\n")

def get_all_dict():
    ctx_dict = {}
    rep_dict = {}
    with open("../data/reddit3/test.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            ctx = " ".join(line[:-1])
            rep = line[-1]
            if ctx not in ctx_dict:
                ctx_dict[ctx] = len(ctx_dict)
            if rep not in rep_dict:
                rep_dict[rep] = len(rep_dict)
    ctx_dict_sort = sorted(ctx_dict.items(), key=lambda x: x[1])
    rep_dict_sort = sorted(rep_dict.items(), key=lambda x: x[1])
    with open("../data/reddit3/test.rep.dict.txt", "w", encoding="utf-8") as fw:
        for (rep, idx) in rep_dict_sort:
            fw.write(str(idx) + "\t" + rep + "\n")
    with open("../data/reddit3/test.ctx.dict.txt", "w", encoding="utf-8") as fw:
        for (rep, idx) in ctx_dict_sort:
            fw.write(str(idx) + "\t" + rep + "\n")

def get_test_ids():
    ctx_dict = {}
    rep_dict = {}
    with open("../data/reddit3/test.ctx.dict.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            idx = line[0]
            ctx = line[1]
            ctx_dict[ctx] = idx
    with open("../data/reddit3/test.rep.dict.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            idx = line[0]
            rep = line[1]
            rep_dict[rep] = idx
    with open("../data/reddit3/test.txt", "r", encoding="utf-8") as fr:
        with open("../data/reddit3/test.id.txt", "w", encoding="utf-8") as fw:
            for idx, line in enumerate(fr):
                line = line.strip().split("\t")
                ctx = " ".join(line[:-1])
                ctx_id = ctx_dict[ctx]
                rep_id = rep_dict[line[-1]]
                fw.write(ctx_id + "\t" + rep_id + "\n")

def get_train_hard_neg():
    ctx_dict = {}
    rep_dict = {}
    with open("../data/reddit3/test.ctx.dict.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            idx = line[0]
            ctx = line[1]
            ctx_dict[idx] = ctx
    with open("../data/reddit3/test.rep.dict.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            idx = line[0]
            rep = line[1]
            rep_dict[idx] = rep
    with open("../data/reddit3/test.id.txt", "r") as fr1:
        with open("../output/result/reddit3/test.hard_neg.idx.txt", "r") as fr2:
            with open("../output/result/reddit3/test.hard_neg.txt", "w", encoding="utf-8") as fw:
                f1 = fr1.readline().strip()
                f2 = fr2.readline().strip()
                count = 0
                neq_count = 0
                while f1 and f2:
                    f1 = f1.split("\t")
                    ctx = ctx_dict[f1[0]]
                    rep = rep_dict[f1[1]]
                    cand = f2.split()[:9]
                    if len(cand) != 9:
                        neq_count += 1
                        continue
                    cand_rep = [rep_dict[x] for x in cand]
                    fw.write(ctx + "\t" + rep + "\t" + "\t".join(cand_rep) + "\n")
                    f1 = fr1.readline().strip()
                    f2 = fr2.readline().strip()
                    count += 1
    print(count)
    print(neq_count)

def make_small_test():
    with open("../output/result/reddit3/train.hard_neg.txt", "r", encoding="utf-8") as fr:
        with open("../output/result/reddit3/train.hard_neg.small.txt", "w", encoding="utf-8") as fw:
            count = 0
            for line in fr:
                line = line.strip()
                fw.write(line + "\n")
                count += 1
                if count == 50000:
                    break


# def check_result():
#     with open("../output/result/reddit3/test.hard_neg.idx.txt", "r", encoding="utf-8") as fr:
#         for line in fr:
#             line = line.strip().split(" ")
#             if len(line) != 20:
#                 print(line)

# making_ret_set()
# making_ret_set2()
# test_generation_recall()
# making_generated_golden_contrast_dataset()
# insert_generation_into_result()
# get_faiss_retrieval_test_set()
# transfer_sample_to_id()
# make_ft_dataset()
# check_result()
# make_concat_dataset()
# get_all_dict()
# get_test_ids()
# get_train_hard_neg()
# make_small_test()