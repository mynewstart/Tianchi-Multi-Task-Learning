import torch
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

PAD,CLS,SEP='[PAD]','[CLS]','[SEP]'

def build_dataset(config):

    def load_dataset(path,pad_size=32,test=False):  #pad_siz表示一个序列的最大长度
        contents=[]
        with open(path,'r',encoding='UTF-8') as f:
            for line in tqdm(f):
                lin=line.strip()
                if not lin:
                    continue
                if 'NLI' in path:
                    if test:
                        id,content1,content2=lin.split('\t')
                        label=0
                    else:
                        id,content1,content2,label=lin.split('\t')
                    token1=config.tokenizer.tokenize(content1)
                    token2=config.tokenizer.tokenize(content2)
                    len1=len(token1)+2
                    len2=len(token2)
                    token=[CLS]+token1+[SEP]+token2
                    segment_id=[]
                    for i in range(len1):
                        segment_id.append(0)
                    for i in range(len2):
                        segment_id.append(1)
                else:
                    if test:
                        id,content1=lin.split('\t')
                        label=0
                    else:
                        id,content1,label=lin.split('\t')
                    token1=config.tokenizer.tokenize(content1)
                    len1=len(token1)+2
                    token=[CLS]+token1+[SEP]
                    segment_id=[]
                    for i in range(len1):
                        segment_id.append(0)
                mask=[]
                token_ids=config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask=[1]*len(token_ids)+[0]*(pad_size-len(token))
                        token_ids+=([0]*(pad_size-len(token)))
                        segment_id+=([0]*(pad_size-len(segment_id)))
                    else:
                        mask=[1]*pad_size
                        token_ids=token_ids[:pad_size]
                        segment_id=segment_id[:pad_size]
                if 'TNEWS' in path:
                    if test:
                        a=0
                    else:
                        a=int(label)
                        if a>=100 and a<105:
                            a-=100
                        elif a>105 and a<=110:
                            a-=101
                        else:
                            a-=102
                    contents.append((token_ids, int(a), segment_id, mask, 2))
                elif 'EMOTION' in path:
                    if test:
                        contents.append((token_ids, int(label), segment_id, mask, 3))
                    else:
                        contents.append((token_ids, int(config.class2id[label]), segment_id, mask, 3))
                else:
                    contents.append((token_ids,int(label),segment_id,mask,1))

        return contents

    train=load_dataset(config.train_path,config.pad_size)
    test=load_dataset(config.test_path,config.pad_size,True)
    return train,test

"""
_函数名表明这是一个私有函数，不应该去访问他（但是仍然能访问到），只是为了标明
__函数名__表示是一个特殊的函数，例如重载运算符
"""

from torch.nn.parallel import DataParallel
import torch
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply

def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids

        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)

        #print('len(inputs): ', str(len(inputs)))
        #print('self.device_ids[:len(inputs)]', str(self.device_ids[:len(inputs)]))

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        if self.gpu0_bsz == 0:
            replicas = self.replicate(self.module, self.device_ids)
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

        # replicas = self.replicate(self.module, device_ids[:len(inputs)])
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]

        #print('replicas:', str(len(replicas)))

        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids[:len(inputs)])

    def scatter(self, inputs, kwargs, device_ids):

        bsz = inputs[0][0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            print(chunk_sizes)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
            print(chunk_sizes)
        else:
            return super().scatter(inputs, kwargs, device_ids)

        print('bsz: ', bsz)
        print('num_dev: ', num_dev)
        print('gpu0_bsz: ', gpu0_bsz)
        print('bsz_unit: ', bsz_unit)
        print('chunk_sizes: ', chunk_sizes)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)

class DatasetIterator(object):
    def __init__(self,batches,batch_size,device):
        self.batch_size=batch_size
        self.batches=batches
        self.n_batches=len(batches)//batch_size
        self.residue=False #余数是否为整数
        if len(batches)%self.n_batches!=0:
            self.residue=True
        self.index=0
        self.device=device

    def _to_tensor(self,datas):
        x = torch.LongTensor([k[0] for k in datas]).to(self.device)
        y = torch.LongTensor([k[1] for k in datas]).to(self.device)
        segment_ids = torch.LongTensor([k[2] for k in datas]).to(self.device)
        mask = torch.LongTensor([k[3] for k in datas]).to(self.device)
        dataset_labels_id=torch.LongTensor([k[4] for k in datas]).to(self.device)

        return (x,segment_ids,mask),y,dataset_labels_id

    def __next__(self):
        if self.residue and self.index==self.n_batches:
            batches=self.batches[self.index*self.batch_size:len(self.batches)]
            self.index+=1
            batches=self._to_tensor(batches)
            return batches
        elif self.index>=self.n_batches:
            self.index=0
            #batches = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            #self.index += 1
            #batches = self._to_tensor(batches)
            #return batches
            raise StopIteration
        else:
            batches=self.batches[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index+=1
            batches=self._to_tensor(batches)
            return batches
    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches+1
        else:
            return self.n_batches

class DatasetIterator2(Dataset):
    def __init__(self,batches,device):
        self.batches=batches
        self.device=device

    def _to_tensor(self,datas):
        x = torch.LongTensor(datas[0]).cuda() #to(self.device)
        y = torch.LongTensor([datas[1]]).squeeze(0).cuda()#.to(self.device)
        segment_ids = torch.LongTensor(datas[2]).cuda()#.to(self.device)
        mask = torch.LongTensor(datas[3]).cuda()#.to(self.device)
        dataset_labels_id = torch.LongTensor([datas[4]]).squeeze(0).cuda()#.to(self.device)
        return (x,segment_ids,mask),y,dataset_labels_id

    def __getitem__(self, item):
        batches=self.batches[item]
        batches=self._to_tensor(batches)
        return batches

    def __len__(self):
        return len(self.batches)

def build_iterator(data,config):
    iter=DatasetIterator(data,config.batch_size,config.device)
    return iter

def build_iterator2(data,config):
    iter=DatasetIterator2(data, config.device)
    return DataLoader(dataset=iter,batch_size=config.batch_size,shuffle=True)#sampler=DistributedSampler(iter)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
