import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel,BertTokenizer
class Config(object):
    #配置参数
    def __init__(self,dataset):
        self.model_name='bert'
        self.train_path=dataset+'raw_data/OCNLI_train_new.txt'
        self.test_path=dataset+'raw_data/OCNLI_a.txt'
        self.label_path=dataset+'raw_data/OCEMOTION_class.txt'
        self.classed=[x.strip() for x in open(self.label_path,encoding='utf-8').readlines()]
        self.class2id,self.id2class={},{}
        for i in range(len(self.classed)):
            self.class2id[self.classed[i]]=i
        for key,value in self.class2id.items():
            self.id2class[value]=key
        self.save_path=dataset+'model_data/'+self.model_name+'.ckpt'
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 30500  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class2id)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 2 # mini-batch大小
        self.pad_size = 256  # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5  # 学习率
        self.bert_path = dataset+'Roberta-wwm-36/'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 1024
        self.rnn_hidden=1024
        self.num_layers=2
        self.hidden_dropout_prob=0.3

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.bert=BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad=True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)
        #self.lstm2 = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
         #                   bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)
        #self.lstm3 = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
         #                   bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_1 = nn.Linear(config.hidden_size*2, 3)
        self.classifier_2 = nn.Linear(config.hidden_size, 15)
        self.classifier_3 = nn.Linear(config.hidden_size, 7)

    def forward(self,x,dataset_labels):
        context=x[0]
        segment_ids=x[1]
        mask=x[2]
        #返回sequence_output和pooled_output，其中sequence_output表示[bz,max_seq_len,768]而pooled_output表示[bz,1,768]只表示了CLS的embedding结果
        sequence_out,pooled=self.bert(context,segment_ids,attention_mask=mask,output_all_encoded_layers=False)
        pooled=self.dropout(pooled)
        if dataset_labels[0].item()==2:
            logitis=self.classifier_2(pooled)
            #out, _ = self.lstm2(sequence_out)
            #out = self.dropout(out)
            #logitis = self.classifier_2(out[:, -1, :])
        if dataset_labels[0].item()==1:
            #logitis=self.classifier_1(pooled)
            out, _ = self.lstm(sequence_out)
            out = self.dropout(out)
            logitis = self.classifier_1(out[:, -1, :])

        if dataset_labels[0].item()==3:
            logitis=self.classifier_3(pooled)
            #out, _ = self.lstm3(sequence_out)
            #out = self.dropout(out)
            #logitis = self.classifier_3(out[:, -1, :])

        return logitis