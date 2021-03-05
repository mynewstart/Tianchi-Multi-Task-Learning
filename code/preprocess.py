import pandas as pd
import codecs
import re
import json
import sys

def data_clear(file):

    """
    数据预处理
    过滤掉非中文，非英文，非数字符号
    """
    with open(file,'r',encoding='utf-8') as f:
        data=f.readlines()
    text=[]
    for d in data:
        d=d.strip().split('\t')
        if 'NLI' in file:
            content1,content2=d[1],d[2]
            text.append(content1)
        else:
            content1=d[1]
            text.append(content1)

    additional_chars = set()
    for t in list(text) + list(text):
        additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', str(t)))
    #print(len(additional_chars))
    # 一些需要保留的符号
    extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")
    print("extra_chars: ",extra_chars)
    additional_chars = additional_chars.difference(extra_chars)
    print("additional_chars:",additional_chars)
    def stop_words(x):
        try:
            x = x.strip()
        except:
            return ''
        x = re.sub('{IMG:.?.?.?}', '', x)
        x = re.sub('<!--IMG_\d+-->', '', x)
        x = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x)
        x = re.sub('<a[^>]*>', '', x).replace("</a>", "")
        x = re.sub('<P[^>]*>', '', x).replace("</P>", "")
        x = re.sub('<strong[^>]*>', ',', x).replace("</strong>", "")
        x = re.sub('<br>', ',', x)
        x = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x).replace("()", "")
        x = re.sub('\s', '', x)   # 过滤不可见字符，如换行和制表符
        x = re.sub('Ⅴ', 'V', x)

        for wbad in additional_chars:
            x = x.replace(wbad, '')
        return x

    # 清除噪声
    text_new=[]
    for t in text:
        if 'NLI' in file:
            t1=stop_words(t[0])
            t2=stop_words(t[1])
            text_new.append(list(t1)+list(t2))
        else:
            t=stop_words(t)
            text_new.append(t)
    with open(file[:-4]+'_new.txt','w',encoding='utf-8') as f:
        for d in data:
            if 'NLI' in file:
                id, content1,content2,label = d.strip().split('\t')
                content1=stop_words(content1)
                content2=stop_words(content2)
                f.write(id+'\t'+content1+'\t'+content2+'\t'+label+'\n')
            else:
                id, _ ,label = d.strip().split('\t')
                content = text_new[int(id)]
                f.write(id + '\t' + content + '\t' + label + '\n')

if __name__=='__main__':
    ocnli='../user_data/raw_data/OCNLI_train.txt'
    tnews='../user_data/raw_data/TNEWS_train.txt'
    ocemotion='../user_data/raw_data/OCEMOTION_train.txt'
    task=[ocnli,tnews,ocemotion]
    for t in task:
        data_clear(t)
