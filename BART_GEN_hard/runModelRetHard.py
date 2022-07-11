import argparse
import torch
import random
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from retrieval_dataset import FileDataset
from BARTRetOnlyModel2 import RetOnlyModel
from encoding_dataset import FileEncodingDataset
from transformers import AdamW, get_linear_schedule_with_warmup, BartConfig, BartModel, BartTokenizer, BartForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default="D:\\PretrainedModels\\BARTBase\\", type=str, help="")
parser.add_argument("--config_name", default="D:\\PretrainedModels\\BARTBase\\", type=str, help="")
parser.add_argument("--tokenizer_name", default="D:\\PretrainedModels\\BARTBase\\", type=str, help="")
parser.add_argument("--data_dir", default="../output/model/", type=str, help="")
parser.add_argument("--output_dir", default="../output/model/", type=str, help="")
parser.add_argument("--result_dir", default="../output/result/reddit3/", type=str, help="")
parser.add_argument("--dataset", default="reddit3", type=str, help="")
parser.add_argument("--do_train", default=True, type=bool, help="")
parser.add_argument("--do_eval", default=True, type=bool, help="")
parser.add_argument("--evaluate_during_training", default=False, type=bool, help="")
parser.add_argument("--per_gpu_train_batch_size", default=50, type=int, help="")
parser.add_argument("--per_gpu_eval_batch_size", default=50, type=int, help="")
parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="")
parser.add_argument("--weight_decay", default=0.0, type=float, help="")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="")
parser.add_argument("--num_train_epochs", default=8, type=int, help="")
parser.add_argument("--max_steps", default=-1, type=int, help="")
parser.add_argument("--warmup_steps", default=-1, type=int, help="")
parser.add_argument("--logging_steps", default=100, type=int, help="")
parser.add_argument("--save_steps", default=-1, type=int, help="")
parser.add_argument("--seed", default=0, type=int, help="")  
args = parser.parse_args()
args.batch_size = args.per_gpu_train_batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.per_gpu_train_batch_size
args.test_batch_size = args.per_gpu_eval_batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.per_gpu_eval_batch_size

tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name)
# additional_tokens = 2
# tokenizer.add_tokens("[enc]")
# tokenizer.add_tokens("[gen]")
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
    train_data = args.data_dir + "/data/" + args.dataset + "/train.hard_neg.txt"
    test_data = args.data_dir + "/data/" + args.dataset + "/test.hard_neg.small.txt"
    config = BartConfig.from_pretrained(args.config_name)
    bart_encdec = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    # bart_encdec.resize_token_embeddings(len(tokenizer))

    model = RetOnlyModel(bart_encdec, config)
    model_state_dict = torch.load(args.data_dir + "/output/models/bart/ensemble.uncertainty.pt")
    model.load_state_dict(model_state_dict, strict=False)
    
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("* number of parameters: %d" % n_params, flush=True)
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
    if args.warmup_steps < 0:
        args.warmup_steps = len(train_dataloader) // 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
    print("***** Running Training *****", flush=True)
    print("Num Examples = ", len(train_dataset), flush=True)
    print("Num Epochs = ", args.num_train_epochs, flush=True)
    print("Batch Size per GPU = ", args.per_gpu_train_batch_size, flush=True)
    print("Total Train Batch Size = ", args.batch_size, flush=True)
    print("Total Optimization Steps = ", t_total, flush=True)

    best_result = 1e5
    global_step = 0
    for epoch in range(args.num_train_epochs):
        print("\nEpoch ", epoch + 1, "/", args.num_train_epochs, flush=True)
        model.train()
        total_loss = 0.0
        tmp_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            loss = train_step(model, batch)
            loss = loss.mean()
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            if step > 0 and step % args.logging_steps == 0:
                print("Step = {:d}\tLR = {:.6f}\tTotal Loss = {:.6f}".format(step, scheduler.get_last_lr()[0], (total_loss - tmp_loss) / args.logging_steps), flush=True)
                tmp_loss = total_loss
            if step > 0 and step % args.save_steps == 0:
                print("Step = {:d}\tLR = {:.6f}\tStart Evaluation".format(step, scheduler.get_lr()[0]), flush=True)
                best_result = evaluate(model, X_test, best_result)
                model.train()
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        print("Epoch = {:d}\tLoss = {:.6f}".format(epoch + 1, total_loss / len(train_dataloader)), flush=True)
        if args.max_steps > 0 and global_step > args.max_steps:
            break

def train_step(model, train_data):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    total_loss = model.forward(train_data)
    return total_loss

def evaluate(model, X_test, best_result):
    model.eval()
    test_dataset = FileDataset(X_test, tokenizer, dataset=args.dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, drop_last=True)
    all_test_loss = 0.0
    with torch.no_grad():
        for test_data in test_dataloader:
            for key in test_data.keys():
                test_data[key] = test_data[key].to(device)
            total_loss = model.forward(test_data)
            all_test_loss += total_loss.mean().item()
    all_test_loss = all_test_loss / len(test_dataloader)
    if all_test_loss < best_result:
        print("Best Test Loss = {:.6f}".format(all_test_loss), flush=True)
        best_result = all_test_loss
        model_to_save = model.module if hasattr(model, 'module') else model
        # model_to_save.model.save_pretrained(args.output_dir)
        torch.save(model_to_save.state_dict(), args.output_dir + "retonly.hard_neg.ft.pt")
    return best_result

def test_model_retrieval():
    test_data = "../data/" + args.dataset + "/test.retrieval.21.bartgenonly.greedy.txt"
    config = BartConfig.from_pretrained(args.config_name)
    bart = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    model = EnsembleModel(bart, config)
    model_state_dict = torch.load("../output/models/bart/ensemble.pt")
    model.load_state_dict(model_state_dict, strict=True)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    dataset = TestDataset(test_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_pred = []
    y_label = []
    with torch.no_grad():
        epoch_iterator = tqdm(dataloader, leave=False)
        for i, test_data in enumerate(epoch_iterator):   
            for key in test_data.keys():
                test_data[key] = test_data[key].to(device)
            y_pred_test = model.forward(test_data, is_test=True)
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
            y_tmp_label = test_data["ret_labels"].data.cpu().numpy().reshape(-1)
            y_label.append(y_tmp_label)
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()

    with open(args.result_dir + "result.bartgenonly.ensemble.rank.txt", "w", encoding="utf-8") as fw:
        for score, label in zip(y_pred, y_label):
            fw.write(str(score) + '\t' + str(label) + '\n')

def encoding_text():
    config = BartConfig.from_pretrained(args.config_name)
    bart = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    model = RetOnlyModel(bart, config)
    model_state_dict = torch.load(args.data_dir + "/output/models/bart/retonly.hard_neg.pt")
    model.load_state_dict(model_state_dict, strict=True)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    test_data = args.data_dir + "/data/" + args.dataset + "/test.ctx.dict.txt"
    dataset = FileEncodingDataset(test_data, tokenizer, max_len=82)
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
    torch.save(all_test_rep, args.output_dir + "test.ctx.retonly.ft.pt", pickle_protocol=4)

    test_data = args.data_dir + "/data/" + args.dataset + "/test.rep.dict.txt"
    dataset = FileEncodingDataset(test_data, tokenizer, max_len=42)
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
    torch.save(all_test_rep, args.output_dir + "test.rep.retonly.ft.pt", pickle_protocol=4)

if __name__ == '__main__':
    set_seed(args.seed)
    # train_model()
    # test_model_retrieval()
    encoding_text()