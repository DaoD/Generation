import argparse
import torch
import random
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from ensemble_dataset import FileDataset
# from ret_test_dataset import FileTestDataset
# from encoding_dataset import FileEncodingDataset
# from RetMetrics import Metrics
from EnsembleModel import EnsembleModel
from transformers import AdamW, get_linear_schedule_with_warmup, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path",
                    default="D:\\PretrainedModels\\DialoGPT\\",
                    type=str,
                    help="")
parser.add_argument("--config_name",
                    default="D:\\PretrainedModels\\DialoGPT\\",
                    type=str,
                    help="")
parser.add_argument("--tokenizer_name",
                    default="D:\\PretrainedModels\\DialoGPT\\",
                    type=str,
                    help="")
parser.add_argument("--output_dir",
                    default="./output/model/",
                    type=str,
                    help="")
parser.add_argument("--result_dir",
                    default="./output/",
                    type=str,
                    help="")
parser.add_argument("--dataset",
                    default="reddit3",
                    type=str,
                    help="")
parser.add_argument("--do_train",
                    default=True,
                    type=bool,
                    help="")
parser.add_argument("--do_eval",
                    default=True,
                    type=bool,
                    help="")
parser.add_argument("--evaluate_during_training",
                    default=False,
                    type=bool,
                    help="")
parser.add_argument("--per_gpu_train_batch_size",
                    default=4,
                    type=int,
                    help="")
parser.add_argument("--per_gpu_eval_batch_size",
                    default=4,
                    type=int,
                    help="")
parser.add_argument("--gradient_accumulation_steps",
                    default=1,
                    type=int,
                    help="")
parser.add_argument("--learning_rate",
                    default=2e-5,
                    type=float,
                    help="")
parser.add_argument("--weight_decay",
                    default=0.0,
                    type=float,
                    help="")
parser.add_argument("--max_grad_norm",
                    default=1.0,
                    type=float,
                    help="")
parser.add_argument("--num_train_epochs",
                    default=8,
                    type=int,
                    help="")
parser.add_argument("--max_steps",
                    default=-1,
                    type=int,
                    help="")
parser.add_argument("--warmup_steps",
                    default=0,
                    type=int,
                    help="")
parser.add_argument("--logging_steps",
                    default=100,
                    type=int,
                    help="")
parser.add_argument("--save_steps",
                    default=-1,
                    type=int,
                    help="")
parser.add_argument("--seed",
                    default=0,
                    type=int,
                    help="")  
args = parser.parse_args()
args.batch_size = args.per_gpu_train_batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.per_gpu_train_batch_size
args.test_batch_size = args.per_gpu_eval_batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.per_gpu_eval_batch_size

tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
special_tokens_dict = {"cls_token": "<CLS>"}
tokenizer.add_special_tokens(special_tokens_dict)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(args)

def set_seed(seed=args.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train_model():
    train_data = "./data/" + args.dataset + "/train.txt"
    test_data = "./data/" + args.dataset + "/validation.txt"
    config = GPT2Config.from_pretrained(args.config_name)
    dialogpt = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config)
    dialogpt.resize_token_embeddings(len(tokenizer))
    model = EnsembleModel(dialogpt)
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("* number of parameters: %d" % n_params)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data, test_data)

def fit(model, X_train, X_test):
    train_dataset = FileDataset(X_train, tokenizer, dataset=args.dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs
    if args.save_steps < 0:
        args.save_steps = len(train_dataloader) // 5

    # no_decay = ["bias", "LayerNorm.weight"]
    # model = model.module if hasattr(model, "module") else model
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    # ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
    print("***** Running Training *****")
    print("Num Examples = ", len(train_dataset))
    print("Num Epochs = ", args.num_train_epochs)
    print("Batch Size per GPU = ", args.per_gpu_train_batch_size)
    print("Total Train Batch Size = ", args.batch_size)
    print("Total Optimization Steps = ", t_total)

    best_result = 1e5
    global_step = 0
    for epoch in range(args.num_train_epochs):
        print("\nEpoch ", epoch + 1, "/", args.num_train_epochs)
        model.train()
        total_loss, total_gen_loss, total_ret_loss = 0.0, 0.0, 0.0
        tmp_loss, tmp_gen_loss, tmp_ret_loss = 0.0, 0.0, 0.0
        for step, batch in enumerate(train_dataloader):
            gen_loss, ret_loss = train_step(model, batch)
            gen_loss = gen_loss.mean()
            ret_loss = ret_loss.mean()
            loss = gen_loss + ret_loss
            loss.backward()
            total_loss += loss.item()
            total_gen_loss += gen_loss.item()
            total_ret_loss += ret_loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            if step > 0 and step % args.logging_steps == 0:
                print("Step = {:d}\tLR = {:.6f}\tTotal Loss = {:.6f}\tGen Loss = {:.6f}\tRet Loss = {:.6f}".format(step, scheduler.get_last_lr()[0], (total_loss - tmp_loss) / args.logging_steps, (total_gen_loss - tmp_gen_loss) / args.logging_steps, (total_ret_loss - tmp_ret_loss) / args.logging_steps))
                tmp_loss = total_loss
                tmp_gen_loss = total_gen_loss
                tmp_ret_loss = total_ret_loss
            if step > 0 and step % args.save_steps == 0:
                print("Step = {:d}\tLR = {:.6f}\tStart Evaluation".format(step, scheduler.get_lr()[0]))
                best_result = evaluate(model, X_test, best_result)
                model.train()
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        print("Epoch = {:d}\tLoss = {:.6f}".format(epoch + 1, total_loss / len(train_dataloader)))
        if args.max_steps > 0 and global_step > args.max_steps:
            break

def train_step(model, train_data):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    gen_loss, ret_loss = model.forward(train_data)
    return gen_loss, ret_loss

def evaluate(model, X_test, best_result):
    model.eval()
    test_dataset = FileDataset(X_test, tokenizer, dataset=args.dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, drop_last=True)
    all_test_loss = 0.0
    all_gen_loss = 0.0
    with torch.no_grad():
        for test_data in test_dataloader:
            for key in test_data.keys():
                test_data[key] = test_data[key].to(device)
            gen_loss, ret_loss = model.forward(test_data)
            all_test_loss += gen_loss.mean().item() + ret_loss.mean().item()
            all_gen_loss += gen_loss.mean().item()
    all_test_loss = all_test_loss / len(test_dataloader)
    all_gen_loss = all_gen_loss / len(test_dataloader)
    if all_test_loss < best_result:
        perplexity = torch.exp(torch.tensor(all_gen_loss))
        print("Best Test Loss = {:.6f}, Perplexity = {:.6f}".format(all_test_loss, perplexity.item()))
        best_result = all_test_loss
        model_to_save = model.module if hasattr(model, 'module') else model
        # model_to_save.model.save_pretrained(args.output_dir)
        torch.save(model_to_save.state_dict(), args.output_dir + "ensemble.tanh.pt")
    return best_result

def test_model_generation():
    test_data = "../data/" + args.dataset + "/test.4predict.txt"
    config = GPT2Config.from_pretrained(args.config_name)
    dialogpt = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config)
    dialogpt.resize_token_embeddings(len(tokenizer))
    model = EnsembleModel(dialogpt)
    model_state_dict = torch.load("../output/models/ensemble/ensemble.tanh.pt")
    model.load_state_dict(model_state_dict, strict=True)
    model = model.to(device)
    # dialogpt.resize_token_embeddings(len(tokenizer))
    # dialogpt = AutoModelForCausalLM.from_pretrained("./output/models/")
    # model = dialogpt.to(device)
    # model = torch.nn.DataParallel(model)
    model.eval()
    all_samples = []
    with open(test_data, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            context = " ".join(line[:-1]) + " " + tokenizer.eos_token
            response = line[-1]
            all_samples.append((context, response))

    start_idx = 0
    end_idx = start_idx + 512
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.model.config.pad_token_id = model.model.config.eos_token_id

    with open(args.result_dir + "result.test.txt", "w", encoding="utf-8") as fw:
        while start_idx < len(all_samples):
            print(start_idx)
            batch_samples = all_samples[start_idx:end_idx]
            batch_contexts = [x[0] for x in batch_samples]
            batch_responses = [x[1] for x in batch_samples]
            inputs = tokenizer(batch_contexts, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            outputs = model.model.generate(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"].to(device),
                max_length=192,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8
            )
            outputs = outputs.cpu()
            batch_out_sentences = tokenizer.batch_decode(outputs[:, input_ids.shape[-1]:], skip_special_tokens=True)
            for idx, r in enumerate(batch_out_sentences):
                fw.write(r + "\t" + batch_responses[idx] + "\n")
            start_idx = end_idx
            end_idx += 512


def test_model_retrieval():
    test_data = "./data/" + args.dataset + "/test.retrieval.20.txt"
    config = GPT2Config.from_pretrained(args.config_name)
    dialogpt = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config)
    dialogpt.resize_token_embeddings(len(tokenizer))
    model = EnsembleModel(dialogpt)
    model_state_dict = torch.load("./output/models/ensemble/ensemble.tanh.pt")
    model.load_state_dict(model_state_dict, strict=True)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    dataset = FileTestDataset(test_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_pred = []
    y_label = []
    with torch.no_grad():
        epoch_iterator = tqdm(dataloader, leave=False)
        for i, test_data in enumerate(epoch_iterator):   
            for key in test_data.keys():
                test_data[key] = test_data[key].to(device)
            y_pred_test = model.forward(test_data, is_ret_test=True)
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
            y_tmp_label = test_data["ret_labels"].data.cpu().numpy().reshape(-1)
            y_label.append(y_tmp_label)
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()

    with open(args.result_dir + "result.ensemble.rank.txt", "w", encoding="utf-8") as fw:
        for score, label in zip(y_pred, y_label):
            fw.write(str(score) + '\t' + str(label) + '\n')

    metrics = Metrics(args.result_dir + "result.ensemble.rank.txt")
    result = metrics.evaluate_all_metrics()
    print("Best Result: R1: %.4f R2: %.4f R5: %.4f MRR: %.4f" % (result[0], result[1], result[2], result[3]))

def encoding_text():
    test_data = "./data/" + args.dataset + "/test.10w.rep.dict.txt"
    config = GPT2Config.from_pretrained(args.config_name)
    dialogpt = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config)
    dialogpt.resize_token_embeddings(len(tokenizer))
    model = EnsembleModel(dialogpt)
    model_state_dict = torch.load("./output/models/ensemble/ensemble.tanh.pt")
    model.load_state_dict(model_state_dict, strict=True)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    dataset = FileEncodingDataset(test_data, tokenizer, max_len=41)
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    all_test_rep = []
    with torch.no_grad():
        epoch_iterator = tqdm(dataloader, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            for key in test_data.keys():
                test_data[key] = test_data[key].to(device)
            test_rep = model.forward(test_data, is_encoding=True).data.cpu().numpy()
            all_test_rep.extend(test_rep)
    all_test_rep = np.asarray(all_test_rep)
    print(all_test_rep.shape)
    torch.save(all_test_rep, args.output_dir + "test.10w.rep.rep.pt")

if __name__ == '__main__':
    set_seed(args.seed)
    # train_model()
    test_model_generation()
    # test_model_retrieval()
    # encoding_text()