import torch
import time
import numpy as np
import argparse
from importlib import import_module  # 用来导入目录下的配置文件
from sklearn.model_selection import train_test_split
from utils import build_dataset, build_iterator, get_time_dif, build_iterator2, BalancedDataParallel
from train_eval import train, init_network, test
import json
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
import random
from sklearn.model_selection import KFold, StratifiedKFold
import time

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=False, default='bert', help='choose a model: Bert')
parser.add_argument('--mode', type=str, required=True, default='train', help='train or test')
# parser.add_argument("--local_rank", type=int, default=-1, help="number of cpu threads to use during batch generation")
args = parser.parse_args()


def BalanceFold(train_data, n_splits):
    Y = []
    for i in range(len(train_data)):
        Y.append(train_data[i][1])
    sfolder = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    train_data = np.array(train_data)
    Y = np.array(np.array(Y))
    cou = 0
    for train_index, dev_index in sfolder.split(train_data, Y):
        cou += 1
        if cou == 1:
            return train_data[train_index], train_data[dev_index]


def check(data, dev):
    dic = {}
    for i in range(len(data)):
        if data[i][1] not in dic:
            dic[data[i][1]] = 1
        else:
            dic[data[i][1]] += 1
    print(dic)
    dic = {}
    for i in range(len(dev)):
        if dev[i][1] not in dic:
            dic[dev[i][1]] = 1
        else:
            dic[dev[i][1]] += 1
    print(dic)


def Pred(model, config):
    print("Testing...")
    with open("../user_data/tmp_data/B_nli_256.txt") as f:
        test_nli = json.load(f)
    with open("../user_data/tmp_data/B_tnews_256.txt") as f:
        test_news = json.load(f)
    with open("../user_data/tmp_data/B_emotion_256.txt") as f:
        test_emotion = json.load(f)
    print(len(test_nli), len(test_news), len(test_emotion))
    test_iter1 = build_iterator(test_nli, config)
    test_iter2 = build_iterator(test_news, config)
    test_iter3 = build_iterator(test_emotion, config)
    test(config, model, test_iter1, test_iter2, test_iter3)
    print("End..")


if __name__ == '__main__':
    dataset = '../user_data/'  # THU新闻数据集
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    print(config.device)
    """
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    config.device = torch.device("cuda", local_rank)
    """

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    """
    产生伪标签输入，效果没有提示，因此被废弃
    #with open("test_nli_256.txt") as f:
    #    a_nli = json.load(f)
        # f.write(json.dumps(train_data)) #128长度
    #with open("test_news_256.txt") as f:
    #    a_news = json.load(f)
   # with open("test_emotion_256.txt") as f:
    #    a_emotion = json.load(f)
    """
    # train

    model = x.Model(config)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(config.device)

    """
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 5) 封装
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
    """

    # model=BalancedDataParallel(0, model, dim=0).cuda()
    # model.to(config.device)

    if args.mode == 'test':
        config.save_path = '../user_data/model_data/bert-12.31.ckpt'
        Pred(model, config)
    else:
        start_time = time.time()
        print("Loading data...")
        with open("../user_data/tmp_data/train_nli_clear.txt") as f:
            train_nli = json.load(f)
        with open("../user_data/tmp_data/train_news_clear.txt") as f:
            train_news = json.load(f)
        with open("../user_data/tmp_data/train_emotion_clear.txt") as f:
            train_emotion = json.load(f)
        print(len(train_nli), len(train_news), len(train_emotion))

        train_data1, dev_data1 = BalanceFold(train_nli, 20)
        train_data2, dev_data2 = BalanceFold(train_news, 20)
        train_data3, dev_data3 = BalanceFold(train_emotion, 10)
        print("dev size: ", len(dev_data1), len(dev_data2), len(dev_data3))
        train_iter1 = build_iterator2(train_data1, config)
        dev_iter1 = build_iterator2(dev_data1, config)
        train_iter2 = build_iterator2(train_data2, config)
        dev_iter2 = build_iterator2(dev_data2, config)
        train_iter3 = build_iterator2(train_data3, config)
        dev_iter3 = build_iterator2(dev_data3, config)
        print(len(dev_iter1), len(dev_iter2), len(dev_iter3))
        print("len(train_features_1)=", len(train_iter1))
        print("len(train_features_2)=", len(train_iter2))
        print("len(train_features_3)=", len(train_iter3))

        a = []
        for i in range(len(train_iter1)):
            a.append(1)
        for i in range(len(train_iter2)):
            a.append(2)
        for i in range(len(train_iter3)):
            a.append(3)
        random.seed(1234)
		#random.seed(1)
        random.shuffle(a)
        print("len(a)=", len(a))
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        config.save_path = '../user_data/Roberta-wwm-36/bert.ckpt'
        # model.load_state_dict(torch.load(config.save_path))
        train(config, model, a, train_iter1, dev_iter1, train_iter2, dev_iter2, train_iter3, dev_iter3)