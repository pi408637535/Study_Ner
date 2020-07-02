import numpy as np
from utils import *

from tqdm import tqdm


class HMM_NER:
    def __init__(self, char2idx_path, tag2idx_path):
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
        # 初始化A, B, pi为全0
        self.transition = np.zeros([self.tag_size,
                                    self.tag_size])
        self.emission = np.zeros([self.tag_size,
                                  self.vocab_size])
        self.pi = np.zeros(self.tag_size)
        # 偏置, 用来防止log(0)或乘0的情况
        self.epsilon = 1e-8

    def estimate_emission_probs(self, train_dic):
        """
        发射矩阵参数的估计
        estimate p( Observation | Hidden_state )
        :param train_dic:
        :return:
        """
        print("estimating emission probabilities...")
        for dic in tqdm(train_dic):
            for char, tag in zip(dic["text"], dic["label"]):
                self.emission[self.tag2idx[tag],
                              self.char2idx[char]] += 1
        self.emission[self.emission == 0] = self.epsilon
        self.emission /= np.sum(self.emission, axis=1, keepdims=True)


    def estimate_transition_probs(self, train_dic):
        """
        转移矩阵和初始概率的参数估计, 也就是bigram二元模型
        estimate p( Y_t+1 | Y_t )
        :param train_dic:
        :return:
        """
        print("estimating transition and initial probabilities...")
        for dic in tqdm(train_dic):
            for i in range(len(dic["label"]) - 1):
                if i == 0: self.pi[self.tag2idx[ self.idx2tag[i] ]] += 1
                tag_p = self.tag2idx[ dic["label"][i] ]
                tag_q = self.tag2idx[ dic["label"][i+1] ]
                self.transition[tag_p, tag_q] += 1

        self.transition[self.transition == 0] = self.epsilon
        self.transition /= np.sum(self.transition, axis=1, keepdims=True)

        self.pi[self.pi == 0] = self.epsilon
        self.pi /= np.sum(self.pi)

    def fit(self, train_dic_path):
        """
        fit用来训练HMM模型
        :param train_dic_path: 训练数据目录
        """
        print("initialize training...")
        train_dic = load_data(train_dic_path)
        # 估计转移概率矩阵, 发射概率矩阵和初始概率矩阵的参数
        self.estimate_transition_probs(train_dic)
        self.estimate_emission_probs(train_dic)
        # take the logarithm
        # 取log防止计算结果下溢
        self.pi = -np.log(self.pi)
        self.transition = -np.log(self.transition)
        self.emission = -np.log(self.emission)

    def viterbi_decode(self, text):

        pi = [ [-1 for j in range(len(text)) ] for i in range(self.tag_size) ]

        f = [ [0 for j in range(len(text)) ] for i in range(self.tag_size) ]

        #init & condition
        for i in range(self.tag_size):
            f[i][0] = self.pi[i] + self.emission[i][self.char2idx[text[0]] ]

        for j in range(1, len(text)): #j timestep
            for i in range( self.tag_size ): # i tag_size
               # f[i][j] = max([ self.transition[i][j] + self.emission[i][self.char2idx[text[j]]  ] ])
                f[i][j] = np.max([ self.transition[:][i] + self.emission[i][self.char2idx[text[j]]]  ])
                pi[i][j] = np.argmax([ self.transition[:][i] + self.emission[i][self.char2idx[text[j]]]  ])

        timestep = len(text)-1
        path = []
        last_tag = np.argmax( np.array( f[-1][:]))

        path.append(last_tag)
        x = len(text)-1
        y = last_tag
        for k in range(1,timestep):
            x -= 1
            y = pi[y][x]
            path.append(pi[y][x])


        data = [ self.idx2tag[ele] for ele in path[::-1] ]
        print(data)

    def predict(self, text):
        if len(text) == 0:
            raise NotImplementedError("输入文本为空!")
        best_tag_id = self.viterbi_decode(text)

if __name__ == '__main__':
    model = HMM_NER(char2idx_path="./dicts/char2idx.json",
                    tag2idx_path="./dicts/tag2idx.json")
    #model.fit("./corpus/train_data.txt")

    model.fit("./corpus/train_data_text.txt")
    model.predict("我在中国吃美国的面包")