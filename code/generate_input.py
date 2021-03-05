import torch
import time
import numpy as np
import argparse
from importlib import import_module #用来导入目录下的配置文件
from sklearn.model_selection import train_test_split
from utils import build_dataset,build_iterator,get_time_dif,build_iterator2,BalancedDataParallel
from train_eval import train,init_network,test
import json
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
import random
from sklearn.model_selection import KFold,StratifiedKFold
import time

parser=argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model',type=str,required=False,default='bert',help='choose a model: Bert')
args=parser.parse_args()

if __name__=='__main__':
    dataset='../user_data/'  #THU新闻数据集
    model_name=args.model
    x=import_module('models.'+model_name)
    config=x.Config(dataset)
    print(config.device)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time=time.time()
    print("Loading data...")
    train_path=['../user_data/raw_data/OCNLI_train_new.txt','../user_data/raw_data/TNEWS_train_new.txt','../user_data/raw_data/OCEMOTION_train_new.txt']
    test_path=['../user_data/raw_data/OCNLI_a.txt','../user_data/raw_data/TNEWS_a.txt','../user_data/raw_data/OCEMOTION_a.txt']
    train_output=["../user_data/tmp_data/train_nli_clear.txt","../user_data/tmp_data/train_news_clear.txt","../user_data/tmp_data/train_emotion_clear.txt"]
    test_output=['../user_data/tmp_data/test_nli256.txt','../user_data/tmp_data/test_news256.txt','../user_data/tmp_data/test_emotion256.txt']

    for i in range(0,len(train_path)):
        config.train_path=train_path[i]
        config.test_path=test_path[i]
        train_data,test_data=build_dataset(config)
        with open(train_output[i],'w') as f:
             f.write(json.dumps(train_data))
        with open(test_output[i],'w') as f:
             f.write(json.dumps(test_data))
    print("End..")
