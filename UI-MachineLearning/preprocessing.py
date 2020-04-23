import numpy as np
import pandas as pd
import jieba
import re
from ml_algorithm import BayesNB

class NLPs():

    def __init__(self, eval_mess):
        self._eval_mess = eval_mess
        self._vocabList = None
        self._vocab_idx = None
        self._data = []
        self.X = None
        self.y = None
        self._bayes = BayesNB()
        self._trained = False

    def add_data(self, text, label):
        """添加数据"""
        self._data.append((text, label))


    def get_label_stat(self):
        cnts = np.zeros(len(self._eval_mess)).astype(np.int32)
        for i in range(len(self._data)):
            idx = self._data[i][1]
            cnts[idx] += 1
        return cnts

    def _replace_all_punctuation(self, text):
        """去除标点符号"""
        return re.sub('\W+', '', text).replace("_", '')  # \W+ 替换掉除了数字字母下划线，所以额外替换下划线


    def train(self):
        """训练模型"""
        # 分词, 构建词库
        self._vocabList = set()
        xlist, ylist = [], []
        for item in self._data:
            word_list = self._cut(text=self._replace_all_punctuation(item[0]))      # 去除标点符号 并 分词
            xlist.append(word_list)
            ylist.append(item[1])
            self._vocabList = self._vocabList | set(word_list)      # 构建词库
        self._vocabList = list(self._vocabList)

        # 构建词库字典 vocab -- index
        self._vocab_idx = dict()
        for i, w in enumerate(self._vocabList):
            self._vocab_idx[w] = i

        # 编码
        self.X = np.zeros((len(xlist), len(self._vocabList)))
        for i in range(len(xlist)):
            self.X[i,:] = self._wordlist2vec(xlist[i])
        self.y = np.array(ylist)

        print('X shape', self.X.shape)
        print('y shape', self.y.shape)

        # 贝叶斯学习
        self._bayes.fit(self.X, self.y)
        self._trained = True


    def predict(self, X):
        """贝叶斯分类"""
        if self._trained == False:
            raise Exception("Can't classify because no bayes study.")
        else:
            return self._bayes.predict(X)


    # 将一篇文章分词，返回分词结果
    def _cut(self, text, onStopwords=False):
        cutlist = jieba.lcut(text)
        return cutlist

    # 将词组转换成向量
    def _wordlist2vec(self, wordlist):
        vec = np.zeros(len(self._vocabList))        # 默认为0
        for word in wordlist:
            if word in self._vocabList:
                vec[self._vocab_idx[word]] += 1
            else:
                # print('%s not in our vocabList.' % word)
                pass
        return vec



    def doc2vec(self, text):
        """用于测试文本转换成向量，不需要更新词库"""
        # 分词 并 去掉标点符号
        wlist = self._cut(self._replace_all_punctuation(text))

        # 编码
        return self._wordlist2vec(wlist)
