# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 10:54
# @Author  : piguanghua
# @FileName: crf.py
# @Software: PyCharm

from sklearn_crfsuite import CRF
#from utils import sent2features
from utils import *
from tqdm import tqdm

class CRFModel(object):
    def __init__(self,
                 algorithm='lbfgs',
                 c1=0.1,
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False
                 ):

        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)


    def train(self, sentences, tag_lists):
        features = [sent2features(s) for s in sentences]
        self.model.fit(features, tag_lists)

    def test(self, sentences):
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists


class TrainCRF():
    def __init__(self, char2idx_path, tag2idx_path,
                 algorithm='lbfgs',
                 c1=0.1,
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False):

        # 载入一些字典
        # char2idx: 字 转换为 token
        self.char2idx = load_dict(char2idx_path)
        # tag2idx: 标签转换为 token
        self.tag2idx = load_dict(tag2idx_path)
        # idx2tag: token转换为标签
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        # 初始化隐状态数量(实体标签数)和观测数量(字数)
        self.tag_size = len(self.tag2idx)
        self.vocab_size = max([v for _, v in self.char2idx.items()]) + 1
        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)



    def train_crf(self,train_dic_path):
        train_dic = load_data(train_dic_path)
        features = []
        labels = []
        for dic in tqdm(train_dic):
            features.append( sent2features(dic["text"]) )
            labels.append(dic["label"])

        self.model.fit(features, labels)

    def predict(self, setence):
        features = [sent2features(s) for s in setence]
        pred_tag_lists = self.model.predict(features)
        print(pred_tag_lists)


if __name__ == '__main__':


    model = TrainCRF(char2idx_path="../dicts/char2idx.json",
                    tag2idx_path="../dicts/tag2idx.json")
    model.train_crf("../corpus/train_data.txt")
    model.predict("我们变而以书会友，以书结缘，把欧美、港台流行的食品类图谱。画册、工具书汇集一堂。")
