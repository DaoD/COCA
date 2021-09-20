import argparse
import random
import numpy as np
import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from BertSessionSearch import BertSessionSearch
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
                    default=128,
                    type=int,
                    help="The batch size.")
parser.add_argument("--per_gpu_test_batch_size",
                    default=128,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=1e-4,
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
                    default="../BERT/BertModel/",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--pretrain_model_path",
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
result_path = "./output/" + args.task + "/"
args.save_path += BertSessionSearch.__name__ + "." +  args.task
args.log_path += BertSessionSearch.__name__ + "." + args.task + ".log"
score_file_prefix = result_path + BertSessionSearch.__name__ + "." + args.task
args.score_file_path = score_file_prefix + "." +  args.score_file_path
args.score_file_pre_path = score_file_prefix + "." +  args.score_file_pre_path

logger = open(args.log_path, "a")
device = torch.device("cuda:0")
print(args)
logger.write("\nHyper-parameters:\n")
args_dict = vars(args)
for k, v in args_dict.items():
    logger.write(str(k) + "\t" + str(v) + "\n")

if args.task == "aol":
    train_data = "./data/aol/train_line.txt"
    test_data = "./data/aol/dev_line.txt"
    predict_data = "./data/aol/test_line.txt"
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 3
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[term_del]")
    tokenizer.add_tokens("[sent_del]")
elif args.task == "tiangong":
    train_data = "./data/tiangong/train.point.txt"
    test_data = "./data/tiangong/dev.point.txt"
    predict_last_data = "./data/tiangong/test.point.lastq.txt"
    predict_pre_data = "./data/tiangong/test.point.preq.txt"
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 4
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[empty_d]")
    tokenizer.add_tokens("[term_del]")
    tokenizer.add_tokens("[sent_del]")
else:
    assert False

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_model():
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size + additional_tokens)
    model_state_dict = torch.load(args.pretrain_model_path)
    bert_model.load_state_dict({k.replace('bert_model.', ''):v for k, v in model_state_dict.items()}, strict=False)
    model = BertSessionSearch(bert_model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data, test_data)

def train_step(model, train_data, bce_loss):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    y_pred = model.forward(train_data)
    batch_y = train_data["labels"]
    loss = bce_loss(y_pred, batch_y)
    return loss

def fit(model, X_train, X_test):
    train_dataset = FileDataset(X_train, 128, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total) * 0, num_training_steps=t_total)
    one_epoch_step = len(train_dataset) // args.batch_size
    bce_loss = torch.nn.BCEWithLogitsLoss()
    best_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
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
                best_result = evaluate(model, X_test, bce_loss, best_result)
                model.train()
            avg_loss += loss.item()
        cnt = len(train_dataset) // args.batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        best_result = evaluate(model, X_test, bce_loss, best_result)
    logger.close()

def evaluate(model, X_test, bce_loss, best_result, X_test_preq=None, is_test=False):
    if args.task == "aol":
        y_pred, y_label = predict(model, X_test)
        metrics = Metrics(args.score_file_path, segment=50)
    elif args.task == "tiangong":
        if is_test:
            y_pred, y_label, y_pred_pre, y_label_pre = predict(model, X_test, X_test_preq)
            metrics_pre = Metrics(args.score_file_pre_path, segment=10)
            with open(args.score_file_pre_path, 'w') as output:
                for score, label in zip(y_pred_pre, y_label_pre):
                    output.write(str(score) + '\t' + str(label) + '\n')
            result_pre = metrics_pre.evaluate_all_metrics()
        else:
            y_pred, y_label = predict(model, X_test)
        metrics = Metrics(args.score_file_path, segment=10)

    with open(args.score_file_path, 'w') as output:
        for score, label in zip(y_pred, y_label):
            output.write(str(score) + '\t' + str(label) + '\n')
            
    result = metrics.evaluate_all_metrics()

    if not is_test and result[0] + result[1] + result[2] + result[3] + result[4] + result[5] > best_result[0] + best_result[1] + best_result[2] + best_result[3] + best_result[4] + best_result[5]:
        best_result = result
        print("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        logger.write("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f \n" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        logger.flush()
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)
    if is_test:
        print("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (result[0], result[1], result[2], result[3], result[4], result[5]))
        if args.task == "tiangong":
            print("Previsou Query Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (result_pre[0], result_pre[1], result_pre[2], result_pre[3], result_pre[4], result_pre[5]))
    return best_result

def predict(model, X_test, X_test_pre=None):
    model.eval()
    test_dataset = FileDataset(X_test, 128, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_pred = []
    y_label = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, ncols=120, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            y_pred_test = model.forward(test_data)
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
            y_tmp_label = test_data["labels"].data.cpu().numpy().reshape(-1)
            y_label.append(y_tmp_label)
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()

    if args.task == "tiangong" and X_test_pre != None:
        test_dataset = FileDataset(X_test_pre, 128, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
        y_pred_pre = []
        y_label_pre = []
        with torch.no_grad():
            epoch_iterator = tqdm(test_dataloader, ncols=120, leave=False)
            for i, test_data in enumerate(epoch_iterator):
                with torch.no_grad():
                    for key in test_data.keys():
                        test_data[key] = test_data[key].to(device)
                y_pred_test = model.forward(test_data)
                y_pred_pre.append(y_pred_test.data.cpu().numpy().reshape(-1))
                y_tmp_label = test_data["labels"].data.cpu().numpy().reshape(-1)
                y_label_pre.append(y_tmp_label)
        y_pred_pre = np.concatenate(y_pred_pre, axis=0).tolist()
        y_label_pre = np.concatenate(y_label_pre, axis=0).tolist()
        return y_pred, y_label, y_pred_pre, y_label_pre
    else:
        return y_pred, y_label

def test_model():
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    model = BertSessionSearch(bert_model)
    model.bert_model.resize_token_embeddings(model.bert_model.config.vocab_size + additional_tokens)
    model_state_dict = torch.load(args.save_path)
    model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    if args.task == "aol":
        evaluate(model, predict_data, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], is_test=True)
    elif args.task == "tiangong":
        evaluate(model, predict_last_data, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], X_test_preq=predict_pre_data, is_test=True)

if __name__ == '__main__':
    set_seed()
    if args.is_training:
        train_model()
        print("start test...")
        test_model()
    else:
        test_model()
