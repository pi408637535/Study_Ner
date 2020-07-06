# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 10:54
# @Author  : piguanghua
# @FileName: utils.py
# @Software: PyCharm


def word2features(sent, i):
    """抽取单个字的特征"""
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i-1]
    next_word = "</s>" if i == (len(sent)-1) else sent[i+1]
    # 使用的特征：
    # 前一个词，当前词，后一个词，
    # 前一个词+当前词， 当前词+后一个词
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word+word,
        'w:w+1': word+next_word,
        'bias': 1
    }
    return features


def sent2features(sent):
    """抽取序列特征"""
    return [word2features(sent, i) for i in range(len(sent))]

import json

def load_dict(path):
    """
    加载字典
    :param path:
    :return:
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_data(path):
    """
    读取txt文件, 加载训练数据
    :param path:
    :return:
    [{'text': ['当', '希', '望', ...],
     'label': ... }, {...}, ... ]
    """
    with open(path, "r", encoding="utf-8") as f:
        return [eval(i) for i in f.readlines()]