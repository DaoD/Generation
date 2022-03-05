import argparse
import random
import numpy as np
import torch
import logging
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from BartGAR import BartGAR
from transformers import AdamW, get_linear_schedule_with_warmup, BartConfig, BartModel, BartTokenizer, BartForConditionalGeneration
from Metrics import Metrics
from GARDataset import GARDataset
from GARTestDataset import GARTestDataset
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path",
                    default="D:\\PretrainedModels\\BARTBase\\",
                    type=str,
                    help="")
parser.add_argument("--config_name",
                    default="D:\\PretrainedModels\\BARTBase\\",
                    type=str,
                    help="")
parser.add_argument("--tokenizer_name",
                    default="D:\\PretrainedModels\\BARTBase\\",
                    type=str,
                    help="")
parser.add_argument("--is_training",
                    default=True,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--per_gpu_batch_size",
                    default=16,
                    type=int,
                    help="The batch size.")
parser.add_argument("--per_gpu_test_batch_size",
                    default=16,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=5e-4,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--task",
                    default="personachat",
                    type=str,
                    help="Task")
parser.add_argument("--file_suffix",
                    default="self_original",
                    type=str,
                    help="Task")
parser.add_argument("--epochs",
                    default=19,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./model/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--pretrain_model_path",
                    default="./model/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="./output/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--log_path",
                    default="./log/",
                    type=str,
                    help="The path to save log.")
args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.per_gpu_batch_size
args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.per_gpu_test_batch_size
if args.task == "cmudog":
    args.save_path += args.task + "." + BartGAR.__name__ + ".onlyret.pt"
    args.score_file_path += BartGAR.__name__ + ".onlyret.txt"
    args.log_path += args.task + "." + BartGAR.__name__ + ".onlyret.log"
else:
    args.save_path += args.task + "." + args.file_suffix + "." + BartGAR.__name__ + ".pt"
    args.score_file_path += BartGAR.__name__ + "." + args.file_suffix + ".txt"
    args.log_path += args.task + "." + args.file_suffix + "." + BartGAR.__name__  + ".log"

tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name)
additional_tokens = 2
tokenizer.add_tokens("[eot]")
tokenizer.add_tokens("[eop]")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(args)

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train_model():
    train_data = "../data/" + args.task + "_processed/train_" + args.file_suffix + "_next.txt"
    test_data = "../data/" + args.task + "_processed/processed_test_" + args.file_suffix + ".single.txt"
    config = BartConfig.from_pretrained(args.config_name)
    bart_model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    bart_model.resize_token_embeddings(len(tokenizer))
    model = BartGAR(bart_model, config)

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('* number of parameters: %d' % n_params)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data, test_data)

def train_step(model, train_data, ce):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    batch_y = train_data["labels"]
    pred, gen_loss = model.forward(train_data)
    loss = ce(pred, batch_y)
    return loss, gen_loss

def fit(model, X_train, X_test):
    train_dataset = GARDataset(X_train, tokenizer, dataset=args.task)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    ce = torch.nn.BCEWithLogitsLoss()
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    one_epoch_step = len(train_dataset) // args.batch_size
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(one_epoch_step * 0.1), num_training_steps=t_total)
    best_result = [0.0, 0.0, 0.0, 0.0]

    # batch = train_dataset.__getitem__(19)
    # print(batch['input_ids'])
    # print(batch['attention_mask'])
    # print(batch['segment_ids'])
    # print(batch['labels'])
    # assert False
    patience = 0
    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        avg_loss = 0.0
        model.train()
        for i, training_data in enumerate(train_dataloader):
            ret_loss, gen_loss = train_step(model, training_data, ce)
            ret_loss = ret_loss.mean()
            gen_loss = gen_loss.mean()
            loss = ret_loss
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']

            if i > 0 and i % (one_epoch_step // 5) == 0:
                print("step: {:d} lr:{:.6f} ret_loss:{:.6f} gen_loss:{:.6f}".format(i, args.learning_rate, ret_loss.item(), gen_loss.item()))
                best_result, patience = evaluate(model, X_test, best_result, patience)
                model.train()
            
            avg_loss += loss.item()
        cnt = len(train_dataset) // args.batch_size + 1
        # tqdm.write("Average loss:{:.6f} Average mse loss:{:.6f}".format(avg_loss / cnt, avg_mse_loss / cnt))
        print("Average loss loss:{:.6f}".format(avg_loss / cnt))
        best_result, patience = evaluate(model, X_test, best_result, patience)
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
            args.learning_rate = param_group['lr']

def evaluate(model, X_test, best_result, patience, is_test=False):
    y_pred, y_label = predict(model, X_test)
    metrics = Metrics(args.score_file_path)

    with open(args.score_file_path, 'w') as output:
        for score, label in zip(y_pred, y_label):
            output.write(str(score) + '\t' + str(label) + '\n')

    result = metrics.evaluate_all_metrics()

    if not is_test and sum(result) > sum(best_result):
        # tqdm.write("save model!!!")
        best_result = result
        print("Best Result: R1: %.4f R2: %.4f R5: %.4f MRR: %.4f" % (best_result[0], best_result[1], best_result[2], best_result[3]))
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)
        patience = 0
    else:
        patience += 1

    if is_test:
        print("Best Result: R1: %.4f R2: %.4f R5: %.4f MRR: %.4f" % (best_result[0], best_result[1], best_result[2], best_result[3]))
    
    return best_result, patience

def predict(model, X_test):
    model.eval()
    test_dataset = GARTestDataset(X_test, tokenizer, dataset=args.task)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_pred = []
    y_label = []
    with torch.no_grad():
        # epoch_iterator = tqdm(test_dataloader, ncols=130, leave=False)
        for i, test_data in enumerate(test_dataloader):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            pred = model.forward(test_data, is_test=True)
            y_pred.append(pred.data.cpu().numpy().reshape(-1))
            batch_label = test_data["labels"].data.cpu().numpy().tolist()
            # y_tmp_label = test_data["labels"].data.cpu().numpy().tolist()
            # y_label_one_hot = np.zeros((len(y_tmp_label), 20), dtype=np.int32)
            # for i in range(len(y_tmp_label)):
            #     y_label_one_hot[i][y_tmp_label[i]] = 1
            # y_label_one_hot = y_label_one_hot.reshape(-1)
            y_label.append(batch_label)

    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()
    return y_pred, y_label

def test_model():
    # model = BartGAR(bert_model, args.qd_insert, args.qd_delete)
    # model.bert_model.resize_token_embeddings(model.bert_model.config.vocab_size + 1)
    # model_state_dict = torch.load(args.save_path)
    # model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
    # model = model.to(device)
    # model = torch.nn.DataParallel(model)
    # evaluate(model, predict_data, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], True)
    pass

if __name__ == '__main__':
    set_seed(0)
    if args.is_training:
        train_model()
    else:
        test_model()
