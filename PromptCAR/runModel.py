import argparse
import random
import numpy as np
import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from BertPromptSS import BertPrefixSS
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
from Trec_Metrics import Metrics
from file_dataset import FileDataset
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--is_training",
                    default=True,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--per_gpu_batch_size",
                    default=100,
                    type=int,
                    help="The batch size.")
parser.add_argument("--per_gpu_test_batch_size",
                    default=128,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--task",
                    default="aol",
                    type=str,
                    help="Task")
parser.add_argument("--epochs",
                    default=3,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./model/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_pre_path",
                    default="score_file.preq.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--bert_model_path",
                    default="/home/yutao_zhu/BertModel/",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--pretrain_model_path",
                    default="",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--data_path",
                    default="",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--log_path",
                    default="./log/",
                    type=str,
                    help="The path to save log.")
args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()
data_path = "./data/" + args.task + "/"
result_path = "./output/" + args.task + "/"
args.save_path += BertPrefixSS.__name__ + "." +  args.task
args.log_path += BertPrefixSS.__name__ + "." + args.task + ".log"
args.score_file_path += BertPrefixSS.__name__ + "." + args.task + ".score.txt"

logger = open(args.log_path, "a")
device = torch.device("cuda:0")
print(args)
logger.write("\nHyper-parameters:\n")
args_dict = vars(args)
for k, v in args_dict.items():
    logger.write(str(k) + "\t" + str(v) + "\n")

if args.task == "aol":
    train_data = args.data_path + "aol/train_line.txt"
    test_data = args.data_path + "aol/test_line.middle.txt"
    predict_data = args.data_path + "aol/test_line.txt"
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 1
    tokenizer.add_tokens("[eos]")
elif args.task == "tiangong":
    train_data = "./data/tiangong/train.point.txt"
    test_last_data = "./data/tiangong/test.point.lastq.txt"
    test_pre_data = "./data/tiangong/test.point.preq.txt"
    predict_data = "./data/tiangong/test.txt"
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 2
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[empty_d]")
else:
    assert False


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


def train_model():
    # load model
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size + additional_tokens)
    model = BertPrefixSS(bert_model)
    if args.pretrain_model_path != "":
        print("load pretrained model...")
        model_state_dict = torch.load(args.pretrain_model_path)
        model.load_state_dict(model_state_dict)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    if args.task == "aol":
        fit(model, train_data, test_data)
    elif args.task == "tiangong":
        fit(model, train_data, test_last_data, test_pre_data)


def train_step(model, train_data, bce_loss):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    y_pred = model.forward(train_data)
    batch_y = train_data["labels"]
    loss = bce_loss(y_pred, batch_y)
    return loss


def fit(model, X_train, X_test, X_test_preq=None):
    train_dataset = FileDataset(X_train, 128, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    one_epoch_step = len(train_dataset) // args.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(one_epoch_step * 0.1), num_training_steps=t_total)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    best_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_result_pre = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # batch = train_dataset.__getitem__(19)
    # print(batch['input_ids'])
    # print(batch['attention_mask'])
    # print(batch['token_type_ids'])
    # print(batch['labels'])
    # assert False

    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
        # epoch_iterator = tqdm(train_dataloader, ncols=120)
        for i, training_data in enumerate(train_dataloader):
            loss = train_step(model, training_data, bce_loss)
            loss = loss.mean()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']

            if i > 0 and i % (one_epoch_step // 5) == 0:
            # if i > 0 and i % 10 == 0:
                if args.task == "aol":
                    best_result = evaluate(model, X_test, bce_loss, best_result)
                elif args.task == "tiangong":
                    best_result = evaluate(model, X_test, bce_loss, best_result, X_test_preq=X_test_preq, best_result_pre=best_result_pre)
                model.train()

            avg_loss += loss.item()

        cnt = len(train_dataset) // args.batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        if args.task == "aol":
            best_result = evaluate(model, X_test, bce_loss, best_result)
        elif args.task == "tiangong":
            best_result = evaluate(model, X_test, bce_loss, best_result, X_test_preq=X_test_preq, best_result_pre=best_result_pre)
    # logger.close()


def evaluate(model, X_test, bce_loss, best_result, X_test_preq=None, best_result_pre=None, is_test=False):
    if args.task == "aol":
        y_pred, y_label = predict(model, X_test)
        metrics = Metrics(args.score_file_path, segment=50)
    elif args.task == "tiangong":
        y_pred, y_label, y_pred_pre, y_label_pre = predict(model, X_test, X_test_preq)
        metrics = Metrics(args.score_file_path, segment=10)
        metrics_pre = Metrics(args.score_file_pre_path, segment=10)
        with open(args.score_file_pre_path, 'w') as output:
            for score, label in zip(y_pred_pre, y_label_pre):
                output.write(str(score) + '\t' + str(label) + '\n')
        result_pre = metrics_pre.evaluate_all_metrics()

    with open(args.score_file_path, 'w') as output:
        for score, label in zip(y_pred, y_label):
            output.write(str(score) + '\t' + str(label) + '\n')
            
    result = metrics.evaluate_all_metrics()

    if not is_test and sum(result[2:]) > sum(best_result[2:]):
        # tqdm.write("save model!!!")
        best_result = result
        # tqdm.write("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        print("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        logger.write("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f \n" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        logger.flush()
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)
    
    if not is_test and args.task == "tiangong" and sum(result_pre) > sum(best_result_pre):
        best_result_pre = result_pre
        # tqdm.write("Previsou Query - Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: " "%.4f" % (result_pre[0], result_pre[1], result_pre[2], result_pre[3], result_pre[4], result_pre[5]))
        print("Previsou Query - Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: " "%.4f" % (result_pre[0], result_pre[1], result_pre[2], result_pre[3], result_pre[4], result_pre[5]))
        logger.write("Previsou Query - Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f \n" % (result_pre[0], result_pre[1], result_pre[2], result_pre[3], result_pre[4], result_pre[5]))
        logger.flush()
    
    if is_test:
        print("Best Result on Test: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (result[0], result[1], result[2], result[3], result[4], result[5]))
        logger.write("Best Result on Test: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f \n" % (result[0], result[1], result[2], result[3], result[4], result[5]))
    
    return best_result


def predict(model, X_test, X_test_pre=None):
    model.eval()
    test_dataset = FileDataset(X_test, 128, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_pred = []
    y_label = []
    with torch.no_grad():
        for i, test_data in enumerate(test_dataloader):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            y_pred_test = model.forward(test_data)
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
            y_tmp_label = test_data["labels"].data.cpu().numpy().reshape(-1)
            y_label.append(y_tmp_label)
            # batch_size = test_data["labels"].size(0)
            # pbar.update(batch_size)
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()

    if args.task == "tiangong":
        test_dataset = FileDataset(X_test_pre, 128, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
        y_pred_pre = []
        y_label_pre = []
        test_iterator = tqdm(test_dataloader, leave=False, ncols=120)
        with torch.no_grad():
            for i, test_data in enumerate(test_iterator):
                with torch.no_grad():
                    for key in test_data.keys():
                        test_data[key] = test_data[key].to(device)
                y_pred_test = model.forward(test_data)
                y_pred_pre.append(y_pred_test.data.cpu().numpy().reshape(-1))
                y_tmp_label = test_data["labels"].data.cpu().numpy().reshape(-1)
                y_label_pre.append(y_tmp_label)
                # batch_size = test_data["labels"].size(0)
                # pbar.update(batch_size)
        y_pred_pre = np.concatenate(y_pred_pre, axis=0).tolist()
        y_label_pre = np.concatenate(y_label_pre, axis=0).tolist()
        return y_pred, y_label, y_pred_pre, y_label_pre
    else:
        return y_pred, y_label


def test_model():
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    model = BertPrefixSS(bert_model)
    model.bert_model.resize_token_embeddings(model.bert_model.config.vocab_size + additional_tokens)
    model_state_dict = torch.load(args.save_path)
    model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    if args.task == "aol":
        evaluate(model, predict_data, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], is_test=True)
    elif args.task == "tiangong":
        pass


if __name__ == '__main__':
    set_seed()
    if args.is_training:
        train_model()
        print("start test...")
        test_model()
    else:
        test_model()