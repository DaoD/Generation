import linecache
import torch 
import numpy as np
import random

def make_sent_level_data():
    sent_dict = {}
    with open("../data/cmudog_processed/processed_train_self_original.txt", "r", encoding="utf8") as fr:
        with open("../data/cmudog_processed/processed_train_self_original.knowledge_sentence.txt", "w", encoding="utf8") as fw:
            for line in fr:
                line = line.strip().split("\t")
                documents = line[3]
                sentences = documents.split("|")
                for sentence in sentences:
                    if sentence not in sent_dict:
                        sent_dict[sentence] = len(sent_dict)
            sorted_dict = sorted(sent_dict.items(), key=lambda x: x[1])
            for (s, idx) in sorted_dict:
                fw.write(str(idx) + "\t" + s + "\n")                        

def make_both_data_for_training():
    with open("../data/personachat_processed/processed_train_self_original.txt", "r", encoding="utf-8") as fr:
        with open("../data/personachat_processed/more/processed_train_self_original.both_turn.txt", "w", encoding="utf-8") as fw:
            for line in fr:
                line = line.strip().split("\t")
                context = (line[0] + " ").split(" _eos_ ")[:-1]
                documents = line[4]
                label = int(line[2])
                if len(context) > 1:
                    new_context = context[:-1]
                    new_context = " _eos_ ".join(new_context)
                    new_context += " _eos_"
                    fw.write(new_context + "\t" + line[1] + "\t" + line[2] + "\t" + line[3] + "\t" + line[4] + "\n")
                fw.write("\t".join(line) + "\n")

def get_dict():
    ctx_dict = {}
    rep_dict = {}
    with open("../data/personachat_processed/processed_train_self_original.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            context = (line[0] + " ").split(' _eos_ ')[:-1]
            context = " [eot] ".join(context)
            context += " [eot]"
            documents = line[4].split("|")
            documents = " [eop] ".join(documents)
            documents += " [eop]"
            context = context + " " + documents
            if context not in ctx_dict:
                ctx_dict[context] = len(ctx_dict)

            responses = line[1].split("|")
            for response in responses:
                response += " [eot]"
                if response not in rep_dict:
                    rep_dict[response] = len(rep_dict)
        ctx_dict = sorted(ctx_dict.items(), key=lambda x: x[1])
        rep_dict = sorted(rep_dict.items(), key=lambda x: x[1])

    with open("../data/personachat_processed/more/train_self_original.ctx.txt", "w", encoding="utf-8") as fw:
        for (x, idx) in ctx_dict:
            fw.write(str(idx) + "\t" + x + "\n")
    with open("../data/personachat_processed/more/train_self_original.rep.txt", "w", encoding="utf-8") as fw:
        for (x, idx) in rep_dict:
            fw.write(str(idx) + "\t" + x + "\n")

def make_id_dataset():
    ctx_dict, rep_dict = {}, {}
    with open("../data/personachat_processed/more/train_self_original.ctx.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            ctx_dict[line[1]] = line[0]
    with open("../data/personachat_processed/more/train_self_original.rep.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            rep_dict[line[1]] = line[0]
    
    with open("../data/personachat_processed/processed_train_self_original.txt", "r", encoding="utf-8") as fr:
        with open("../data/personachat_processed/more/processed_train_self_original.id.txt", "w", encoding="utf-8") as fw:
            for line in fr:
                line = line.strip().split("\t")
                context = (line[0] + " ").split(' _eos_ ')[:-1]
                context = " [eot] ".join(context)
                context += " [eot]"
                documents = line[4].split("|")
                documents = " [eop] ".join(documents)
                documents += " [eop]"
                context = context + " " + documents
                ctx_id = ctx_dict[context]

                responses = line[1].split("|")
                label = int(line[2])
                response = responses[label]
                response += " [eot]"
                rep_id = rep_dict[response]
                fw.write(ctx_id + "\t" + rep_id + "\n")
                
def remove_rel_id():
    pairs = []
    with open("../data/personachat_processed/more/processed_train_self_original.id.txt", "r") as fr:
        for line in fr:
            line = line.strip().split("\t")
            pairs.append((line[0], line[1]))
    # sim_dict = {}
    # with open("../../data/reddit3/train.repsim.top110.txt", "r") as fr:
    #     for idx, line in enumerate(fr):
    #         if idx % 500000 == 0:
    #             print(idx)
    #         line = line.strip().split("\t")
    #         sim_dict[idx] = line
    with open("../data/personachat_processed/more/train.rep_sim.top100.txt", "w") as fw:
        for idx, pair in enumerate(pairs):
            if idx % 10000 == 0:
                print(idx)
            ctx_id = pair[0]
            rep_id = pair[1]
            cand_ids = linecache.getline("../data/personachat_processed/more/train_rep_sim.top110.txt", int(rep_id) + 1)
            cand_ids = cand_ids.split("\t")
            if rep_id in cand_ids:
                cand_ids.remove(rep_id)
            cand_ids = cand_ids[:100]
            fw.write(ctx_id + "\t" + rep_id + "\t" + "\t".join(cand_ids) + "\n")
    print(len(pairs))

def present_results():
    rep_dict = {}
    with open("../data/personachat_processed/more/train_self_original.rep.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            rep_dict[line[0]] = line[1]
    with open("../data/personachat_processed/more/train.rep_sim.top100.txt", "r") as fr:
        for line in fr:
            line = line.strip().split("\t")
            for x in line:
                print(rep_dict[x])
            break

def check_result_similarity():
    ctx_dict, rep_dict = {}, {}
    with open("../data/personachat_processed/more/train_self_original.ctx.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            ctx_dict[line[0]] = line[1]
    with open("../data/personachat_processed/more/train_self_original.rep.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            rep_dict[line[0]] = line[1]
    rep_idx_list = list(range(len(rep_dict)))
    ctx_rep = torch.load("../data/personachat_processed/more/BertRS.self_original.train.ctx.pt")
    rep_rep = torch.load("../data/personachat_processed/more/BertRS.self_original.train.rep.pt")
    with open("../data/personachat_processed/more/train.rep_sim.top100.txt", "r") as fr:
        count = 0
        for line in fr:
            line = line.strip().split("\t")
            ctx_id = int(line[0])
            rep_id = int(line[1])
            cand_ids = line[2:]
            # cand_ids = random.sample(rep_idx_list, 50)
            cand_ids = random.sample(cand_ids, 50)
            candidate_rep = np.array([rep_rep[int(x)] for x in cand_ids])
            candidate_rep = torch.FloatTensor(candidate_rep)
            context_rep = torch.FloatTensor(ctx_rep[ctx_id])
            response_rep = torch.FloatTensor(rep_rep[rep_id])
            true_scores = torch.sum(context_rep * response_rep, dim=0)
            candi_scores = torch.einsum("d,nd->n", context_rep, candidate_rep)
            score_threshold = true_scores
            candi_scores = candi_scores - 0.8 * score_threshold
            count += 1
            print(torch.sum(candi_scores < 0))
            candi_scores = candi_scores.masked_fill(candi_scores > 0, float('-inf'))
            top_scores, top_indexs = torch.topk(candi_scores, k=19, dim=0)
            print(top_scores)
            if count == 50:
                break


def transfer_data_format():
    with open("../data/personachat_processed/processed_test_self_original.txt", "r", encoding="utf-8") as fr:
        with open("../data/personachat_processed/more/test.txt", "w", encoding="utf-8") as fw:
            for line in fr:
                line = line.strip().split("\t")
                context = (line[0] + " ").split(" _eos_ ")[:-1]
                responses = line[1].split("|")
                documents = line[4].split("|")
                label = int(line[2])
                for idx, resposne in enumerate(responses):
                    if idx == label:
                        fw.write("1\t" + "\t".join(documents) + " [split] " + "\t".join(context) + "\t" + resposne + "\n")
                    else:
                        fw.write("0\t" + "\t".join(documents) + " [split] " + "\t".join(context) + "\t" + resposne + "\n")

# make_sent_level_data()
# make_both_data_for_training()
# get_dict()
# make_id_dataset()
# remove_rel_id()
# present_results()
# check_result_similarity()
transfer_data_format()