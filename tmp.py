import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam
import pandas as pd
from sklearn.metrics import classification_report
import csv
import json
import os
import random
from tqdm import tqdm

accumulation_steps = 8  # 放大batch_size的大小
# 对抗样本攻击，在embedding层做扰动
alpha1 = [0.335, 0.324, 0.341]
alpha2 = [1.08, 0.30, 0.24, 0.30, 0.23, 0.57, 0.29, 0.35, 0.20, 0.33, 0.35, 0.25, 4.73, 0.42, 0.35]
alpha3 = [0.66, 0.30, 0.22, 4.56, 0.66, 0.61, 2.98]


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=False):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss, self).__init__()
        num_classes = len(alpha)
        print(num_classes)
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        print("loss: ", preds.shape, labels.shape, len(self.alpha))
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds,
                                  dim=1)  # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)  # [4,15]-->[4,15]
        logpt = logpt.gather(1, target)  # [4,15]-->[4,1]
        logpt = logpt.view(-1)
        pt = torch.tensor(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.tensor(at)  # 对应相乘

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                print(name, param)
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                print(name, param)
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model, eps=1., alpha=0.3):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='emb', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        print(name, w.size())
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, a, train_data1, dev_data1, train_data2, dev_data2, train_data3, dev_data3):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 不进行权重衰减的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=0.05,
                         t_total=len(a) * config.num_epochs)  # total number of training steps for the learning
    total_batch = 0  # 记录进行到多少次batch
    dev_best_loss = float('-inf')
    last_impove = 0  # 记录上次验证集loss下降的batch数目
    flag = False
    model.train()
    fgm = FGM(model)
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for step, number in enumerate(a):
            if number == 1: batches = train_data1.__iter__().__next__()
            if number == 2: batches = train_data2.__iter__().__next__()
            if number == 3: batches = train_data3.__iter__().__next__()
            trains, labels, dataset_labels_ids = batches
            outputs = model(trains, dataset_labels_ids)
            loss = F.cross_entropy(outputs, labels)
            # if number==2: loss=loss*1.1
            # if number==3: loss=loss*1.2
            # fgm.attack()
            # loss_adv = F.cross_entropy(model(trains, dataset_labels_ids),labels)
            # loss_adv.backward()
            # fgm.restore()
            # 2.1 loss regularization
            loss = loss / accumulation_steps
            # 2.2 back propagation
            loss.backward()
            # 进行梯度累加
            if ((step + 1) % accumulation_steps == 0):
                optimizer.step()
                model.zero_grad()
            if total_batch % (len(a) // 10) == 0:
                # 输出在训练集和测试集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, macro_f1, dev_loss = evaluate(config, model, dev_data1, dev_data2, dev_data3)
                if macro_f1 > dev_best_loss:
                    dev_best_loss = macro_f1
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_impove = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Val F1: {5:>6.2%},Time: {6} {7}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, macro_f1, time_dif, improve))
                model.train()
            total_batch += 1
            if epoch >= 11 and total_batch - last_impove > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test(config, model, test_iter1, test_iter2, test_iter3):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    predict(config, model, test_iter1, "../prediction_result/ocnli_predict.csv", 1)
    predict(config, model, test_iter2, "../prediction_result/tnews_predict.csv", 2)
    predict(config, model, test_iter3, "../prediction_result/ocemotion_predict.csv", 3)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def predict(config, model, data_iter, file, id):
    y_pred = []
    Len = len(data_iter)
    print("Len", Len, id)
    dic = {}
    if id == 1:
        dic[0] = '0';
        dic[1] = '1';
        dic[2] = '2'
    elif id == 2:
        dic[0] = '100';
        dic[1] = '101';
        dic[2] = '102';
        dic[3] = '103';
        dic[4] = '104';
        dic[5] = '106';
        dic[6] = '107';
        dic[7] = '108';
        dic[8] = '109';
        dic[9] = '110';
        dic[10] = '112';
        dic[11] = '113';
        dic[12] = '114';
        dic[13] = '115';
        dic[14] = '116'
    else:
        dic[0] = 'like';
        dic[1] = 'happiness';
        dic[2] = 'sadness';
        dic[3] = 'fear';
        dic[4] = 'anger';
        dic[5] = 'disgust';
        dic[6] = 'surprise'
    with torch.no_grad():
        for texts, _, dataset_labels_ids in data_iter:
            outputs = model(texts, dataset_labels_ids)
            label = torch.max(outputs, dim=1)[1].cpu().numpy().tolist()
            for i in range(len(label)):
                y_pred.append(dic[label[i]])
    a = []
    for i in range(1, len(y_pred) + 1):
        a.append(i)
    df = pd.DataFrame({'id': a, 'label': y_pred})
    df.to_csv(file, index=False, sep=',')
    csvfile = open(file, 'r')
    file_json = file[:-3] + "json"
    fieldnames = ('id', 'label')
    reader = csv.DictReader(csvfile, fieldnames)
    os.remove(file)
    data_list = [row for row in reader]
    with open(file_json, "w") as f:
        for i in range(1, len(data_list)):
            f.write(json.dumps(data_list[i], ensure_ascii=False) + '\n')


def evaluate(config, model, dev_data1, dev_data2, dev_data3):
    model.eval()
    loss_total = 0
    Len = len(dev_data1) + len(dev_data2) + len(dev_data3)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels, dataset_labels_ids in dev_data1:
            outputs = model(texts, dataset_labels_ids)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc_1 = metrics.accuracy_score(labels_all, predict_all)
    report_1 = classification_report(labels_all, predict_all, digits=4)
    avg_f1_1 = float(report_1.split()[-8])

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels, dataset_labels_ids in dev_data2:
            outputs = model(texts, dataset_labels_ids)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc_2 = metrics.accuracy_score(labels_all, predict_all)
    report_2 = classification_report(labels_all, predict_all, digits=4)
    avg_f1_2 = float(report_2.split()[-8])

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels, dataset_labels_ids in dev_data3:
            outputs = model(texts, dataset_labels_ids)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc_3 = metrics.accuracy_score(labels_all, predict_all)
    report_3 = classification_report(labels_all, predict_all, digits=4)
    avg_f1_3 = float(report_3.split()[-8])
    print(avg_f1_1, avg_f1_2, avg_f1_3)
    return (acc_1 + acc_2 + acc_3) / 3.0, (avg_f1_1 + avg_f1_2 + avg_f1_3) / 3.0, loss_total / Len
