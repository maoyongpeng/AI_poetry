#coding:utf-8
import pickle
import sys
import os
import json
import re
from collections import Counter

import numpy as np
from opencc import OpenCC
from configs import opt_shi,opt_ci
def toList(sen): # 将字符串转换为列表
    return [s for s in sen]

cc=OpenCC("t2s")
def loaddatafromdict(dict_path, category, constrain=None):

    def eraseothers(paras):
        """
        移除不必要的字符
        :param paras: poem
        :return: result
        """
        # 一次性移除括号及其内容，包括中文括号（）、大括号{}和书名号《》，以及方括号[]
        result = re.sub(u"（.*?）|{.*?}|《.*?》|[\[\]]", "", paras)
        #去除方形字符
        result = re.sub(u"[□◻]", "", result)

        # 移除数字和减号
        result = ''.join(s for s in result if s not in '0123456789-')
        # 将连续的句号替换为单个句号
        result = re.sub(u"。。+", u"。", result)
        result=cc.convert(result)
        return result

    def extraJsonInfo(file):
        poems_process = []
        with open(file, 'r',encoding='utf-8') as f:  # 打开文件
            data = json.load(f)

        for poetry in data:
            paragraphs = poetry.get("paragraphs", []) #返回的是一个列表
            pdata = "".join(paragraphs)  # 合并段落
            pdata = eraseothers(pdata)  # 预处理，移除不必要的字符 【】（）《》 0123456789-

            if pdata and all(constrain == len(tr)
                             or len(tr) == 0 for s in paragraphs
                             for tr in re.split(u"[，！。]", s) if
                             constrain is not None):
                poems_process.append(pdata)  # 检查长度约束并添加满足条件的诗歌
        return poems_process

    data = []
    for filename in os.listdir(dict_path): # 遍历文件夹
        if filename.startswith(category): # 判断文件名是否以category(分类)开头
            data.extend(extraJsonInfo(dict_path + filename))
    return data


def build_vocab(data):
    """构建词汇表"""
    word_freq = Counter(''.join(data))  # 使用Counter统计词频，较高的词频分配较低的索引
    words = word_freq.keys()
    word2ix = {word: ix for ix, word in enumerate(words)}

    word2ix['<EOP>'] = len(word2ix)
    word2ix['<START>'] = len(word2ix)
    word2ix['<PAD>'] = len(word2ix)
    ix2word = {ix: word for word, ix in word2ix.items()}
    return word2ix, ix2word


def optimize_get_data(opt):
    """
    获取数据
    :param opt:
    :return:
    data: 截断或者填充后的诗歌数据, list
    word2ix: 词汇表到索引, dict
    ix2word: 索引到词汇表, dict
    """
    if os.path.exists(opt.npz_data) :
        npzfile = np.load(opt.npz_data, allow_pickle=True)
        data=npzfile['data']
        word2ix = npzfile['word2ix'].item()
        ix2word = npzfile['ix2word'].item()
        return data,word2ix,ix2word


    data = loaddatafromdict(dict_path=opt.dict_path, category=opt.category)
    word2ix, ix2word = build_vocab(data)

    # 为每首诗歌加上起始符和终止符
    for i in range(len(data)):
        data[i] = ["<START>"] + list(data[i]) + ["<EOP>"]


    new_data = [[word2ix[word] for word in sentence]
                for sentence in data]
    # print(new_data)
    # 诗歌长度不够opt.maxlen的在后面补空格，超过的opt.maxlen，删除末尾的字符
    pad_data = pad_sequences(new_data, maxlen=opt.maxlen, padding='post', truncating='post', value=len(word2ix) - 1)

    np.savez_compressed(opt.npz_data, data=pad_data, word2ix=word2ix, ix2word=ix2word)
    return pad_data, word2ix, ix2word

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.):

    num_samples = len(sequences)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
if __name__ == '__main__':
    data=loaddatafromdict(opt_shi.dict_path,category='poet.tang')
    # print(data[:5])
    data,word2ix,ix2word=optimize_get_data(opt_ci)
    print(data[:5])
    print(len(data))
    print(len(word2ix)-1)
    print(word2ix['<PAD>'])

    pass

