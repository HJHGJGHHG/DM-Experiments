import time
import torch
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import config
from dataset import MovieRankDataset
from torch.utils.data import DataLoader
from model import MovieLens

args = config.args_initialization()
args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
# args.device = 'cpu'

# --------------- hyper-parameters------------------
user_max_dict = {
    'uid': 6041,  # 6040 users
    'gender': 2,
    'age': 7,
    'job': 21
}

movie_max_dict = {
    'mid': 3953,  # 3883 movies
    'mtype': 18,
    'mword': 5214  # 5214 words
}


def get_time_diff(start_time):
    end_time = time.perf_counter()
    return '总用时: {0}分  {1}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60)


def set_seed(args):
    # 设定种子保证可复现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def load_model(args):
    train_datasets = MovieRankDataset(pkl_file=args.path + 'train.pkl', args=args)
    test_datasets = MovieRankDataset(pkl_file=args.path + 'test.pkl', args=args)
    train_iter = DataLoader(train_datasets, batch_size=args.batch_size, drop_last=True)
    test_iter = DataLoader(test_datasets, batch_size=args.batch_size)
    model = MovieLens(user_max_dict=user_max_dict, movie_max_dict=movie_max_dict, args=args).to(args.device)
    return train_iter, test_iter, model


def train(args, model, train_iter):
    # optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr=args.lr)
    # criterion
    criterion = nn.MSELoss()
    writer = SummaryWriter(logdir='/root/tf-logs')
    for epoch in range(args.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, args.epochs))
        model = model.train()
        losses = []
        for i_batch, sample_batch in enumerate(train_iter):
            user_inputs = sample_batch['user_inputs']
            movie_inputs = sample_batch['movie_inputs']
            target = torch.squeeze(sample_batch['target'].to(args.device))
            
            rank, _, _ = model(user_inputs, movie_inputs)
            loss = criterion(rank, target)
            losses.append(loss.item())
            loss.backward()
            if i_batch % 100 == 0:
                writer.add_scalar('data/loss', loss, i_batch * 20)
                print('Steps: {}\{}  Loss: {}'.format(i_batch, len(train_iter), loss.item()))
            
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        # 一个Epoch训练完毕，输出train_loss
        print('Epoch: {0}   Train Loss: {1:>5.6}'.format(epoch + 1, np.mean(losses)))
    
    writer.export_scalars_to_json("./test.json")
    writer.close()
