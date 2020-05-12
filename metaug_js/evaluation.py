import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import random
import argparse

from load_data import *
from utils import *
from training import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument("--data_prefix", help="prefix for the datasets", type=str, default="phonology")
parser.add_argument("--vocab_size", help="vocab size for the model", type=int, default=34)
parser.add_argument("--emb_size", help="embedding size for the model", type=int, default=10)
parser.add_argument("--hidden_size", help="hidden size for the model", type=int, default=256)
parser.add_argument("--method", help="training method", type=str, default="maml")
parser.add_argument("--lr_inner", help="inner loop learning rate", type=float, default=1.0)
parser.add_argument("--inner_batch_size", help="inner loop batch size", type=int, default=100)
parser.add_argument("--lr_outer", help="outer loop learning rate", type=float, default=0.001)
parser.add_argument("--outer_batch_size", help="outer loop batch size", type=int, default=1)
parser.add_argument("--print_every", help="how many iterations to pass before printing dev accuracy", type=int, default=1000)
parser.add_argument("--patience", help="how many prints to pass before early stopping", type=int, default=5)
parser.add_argument("--save_prefix", help="prefix for saving the weights file", type=str, default="phonology")
parser.add_argument("--eval_technique", help="whether to evaluate on a metalearning test set (meta), i.e., accuracy after one batch, or to train to convergence (converge) with some threshold", type=str, default="meta")
parser.add_argument("--threshold", help="accuracy threshold after which you should early stop", type=float, default=0.9)
parser.add_argument("--by_ranking", help="whether to break down results by ranking", type=str, default="False")
parser.add_argument("--train", help="whether to train the model on each task before evaluating", type=str, default="True")
parser.add_argument("--update_embeddings", help="whether to update the embeddings on each task before evaluating", type=str, default="True")
args = parser.parse_args()

test_set = load_dataset(args.data_prefix + ".test")
model = EncoderDecoder(args.vocab_size,args.emb_size,args.hidden_size)
model.load_state_dict(torch.load(args.save_prefix + ".weights"))

if args.eval_technique == "meta":
    train = args.train == "True"
    if args.by_ranking == "True":
        avg_acc = average_acc_by_ranking(model, test_set, lr_inner=args.lr_inner, batch_size=args.inner_batch_size, train=train)
    else:
        avg_acc = average_acc(model, test_set, lr_inner=args.lr_inner, batch_size=args.inner_batch_size, train=train, update_embeddings=args.update_embeddings=="True")
    print(avg_acc)

elif args.eval_technique == "converge":
    total_iters = 0
    total_test_acc = 0
 
    for task in test_set:
        model_copy = model.create_copy()

        num_iters, dev_acc, test_acc = train_model(model_copy, task, batch_size=args.inner_batch_size, print_every=args.print_every, patience=args.patience, threshold=args.threshold)
        total_iters += num_iters
        total_test_acc += test_acc

    print("Total iters:", total_iters * 1.0 / len(test_set), "Test acc:", total_test_acc * 1.0 / len(test_set))
    

elif args.eval_technique == "converge_by_ranking":
    acc_dict = {}
 
    for task in test_set:
        model_copy = model.create_copy()

        num_iters, dev_acc, test_acc = train_model(model_copy, task, batch_size=args.inner_batch_size, print_every=args.print_every, patience=args.patience, threshold=args.threshold)
        ranking = tuple(task[-1][-1])
        if ranking not in acc_dict:
            acc_dict[ranking] = [0,0]

        acc_dict[ranking][0] += num_iters
        acc_dict[ranking][1] += 1  


    avg_acc_list = []

    for key in acc_dict:
        avg_acc_list.append([key, acc_dict[key][0] / acc_dict[key][1]])


    print(avg_acc_list)
    











