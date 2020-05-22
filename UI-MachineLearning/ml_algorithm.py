import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class KNN():
    """k最近邻算法实现"""

    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.train_num = len(self.X_train)
        self.classes = np.unique(y_train)
        self.classnum = len(self.classes)
        return self

    def predict(self, X_test):
        """使用numpy代替列表实现"""
        n_test = len(X_test)
        y_pre = np.zeros(n_test)
        for ith, item in enumerate(X_test):
            dicts = np.zeros(len(self.X_train))
            for idx, d in enumerate(self.X_train):
                dicts[idx] = (np.linalg.norm(item - d))  # 欧式距离
            indexs = np.argsort(dicts)  # 根据值得索引排序
            cnts = np.zeros(self.classnum)
            for idx in indexs[:self.k]:  # 统计最近k个数据的标签
                tag = self.y_train[idx]
                cnts[tag] += 1
            imax = np.argmax(cnts)
            pre = self.classes[imax]  # 找到周围出现最多的标签
            # print(pre)
            y_pre[ith] = pre  # 记录每一个数据对应预测的分类结果
        return y_pre

    def predict_accuracy(self, X_test, y_test):
        y_pre = self.predict(X_test)  # 正确率
        return (y_pre == y_test).sum() * 1.0 / len(X_test)

    def predict_error_rate(self, X_test, y_test):  # 错分率
        return 1 - self.predict_accuracy(X_test, y_test)


class BayesNB():

    def __init__(self):
        pass

    def fit(self, X, y):
        """针对文本的bayes，假设X[0] = (1, 0, 0, 1), y[0] = 1"""
        self.nSample = len(X)
        self.nFeature = len(X[0])
        self.category = list(set(y))
        self.nCategory = len(self.category)
        self.pPriors = np.zeros(self.nCategory)
        self.matrix = np.ones((self.nCategory, self.nFeature))  # 使用拉普拉斯平滑lambda=1
        self._nWordCate = np.zeros(self.nCategory) + self.nCategory  # K * lambda

        # 统计先验概率
        for i in range(len(self.category)):
            self.pPriors[i] = np.sum(self.category[i] == y) / self.nSample

        # 统计每个词语在对应类别下出现的次数
        for i in range(self.nSample):
            idxCate = np.argwhere(self.category == y[i])
            self.matrix[idxCate, :] += X[i]
            self._nWordCate[idxCate] += np.sum(self.matrix[idxCate, :])

        # 将matrix里面的次数转换成出现的频率 --> log(pMatrix)使数值不至于下溢
        self.logMatrix = np.zeros_like(self.matrix)
        for i in range(len(self.matrix)):
            self.logMatrix[i, :] = np.log(self.matrix[i, :] / self._nWordCate[i])
        return self

    def predict(self, Xtest):
        """使用统计的先验概率pPriors和对应的Xtest判断属于各类的概率，并且取其中的最大值作为分类"""
        '''假设Xtest = [[1, 1, 0, 0],
                        [0, 0, 1, 0]]'''
        y_pred = np.zeros(len(Xtest)).astype(np.int32)

        for r in range(len(Xtest)):  # 每一行数据
            pCates = np.zeros(self.nCategory)
            # 遍历各个分类求得对于每种分类的后验概率
            for i in range(self.nCategory):  # 每一种分类
                pCates[i] = np.sum(self.logMatrix[i] * Xtest[r]) + np.log(self.pPriors[i])
            y_pred[r] = self.category[np.argmax(pCates)]
        return y_pred


class BP:
    """三层BP神经网络"""
    # TODO: 需要修正为softmax的多分类

    def __init__(self, n_input, n_output, lr=0.001, is_auto_lr=False, epoch=200, accuary=0.0001):
        self.X = None
        self.y = None
        self.lr = lr
        self.is_auto_lr = is_auto_lr
        self.epoch = epoch
        self.accuary = accuary
        self.errors = []
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = int(math.sqrt(n_input + n_output) + 10 + 0.5)  # TODO:尝试改为随机数
        self.n_feature = 0
        self.IW = None
        self.HW = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_feature = self.X.shape[1]
        self.IW = np.random.uniform(0.0, 1.0, size=(self.n_hidden, self.n_input + 1)) / self.n_feature
        self.HW = np.random.uniform(0.0, 1.0, size=(self.n_output, self.n_hidden + 1)) / self.n_feature
        self.N = len(self.X)

        for t in range(self.epoch):
            # 精度退出
            # if self._is_quit(t):
            #     break
            # 自动更新学习率模块
            self._auto_lr(t)
            # 整个前向计算
            self._feed_forward()
            # 整个反向传播
            self._back()
            # 计算总误差
            self.errors.append((1.0 / (2)) * np.sum(self._error * self._error))

    def predict(self, X):
        """预测结果，即运行一次前向计算"""
        self.X = X.copy()
        self._feed_forward()

        # 输出层结果作为输出
        return self._out.copy()

    def _feed_forward(self):
        """前向计算"""
        # 隐藏层输入，转置并在第一行添加一行1，作为bias shape=(n1, N)
        self._in = np.row_stack((np.ones(len(self.X)), np.transpose(self.X)))

        # 计算隐藏层结果，并通过激活函数 shape=(n2, N)
        self._hide = self.activation(np.dot(self.IW, self._in))

        # 输出层输入，在第一行添加一行，作为bias shape=(n2+1, N)
        self._hide1 = np.row_stack((np.ones(len(self.X)), self._hide))

        # 计算输出层结果，shape=(n3, N)
        self._out = np.dot(self.HW, self._hide1)

        # 计算误差 shape=(n3, N)
        self._error = np.transpose(self.y) - self._out

    def _back(self):
        """反向传播"""
        # 输出层灵敏度 shape=(n3, N)
        self._sens_out = self._out * (1 - self._out) * self._error

        # 隐藏层灵敏度
        tmp_HW = self.HW[:, 1:]  # 去掉第一列，bias相关
        # tmp_HW需要转置，再与输出层灵敏度点乘， shape=(n2, N) . (n3, N) = (n2, N)
        self._sens_hide = np.dot(np.transpose(tmp_HW), self._sens_out)

        # 更新输出层权重
        self.HW = self.HW + (self.lr / self.N) * np.dot(self._sens_out, np.transpose(self._hide1))

        # 更新隐藏层权重
        self.IW = self.IW + (self.lr / self.N) * np.dot(self._sens_hide, np.transpose(self._in))

    def _is_quit(self, t):
        if t % 10 == 0 and t != 0:
            de = self.errors[-1] - self.errors[-2]
            if abs(de) < self.accuary:
                return True
        return False

    def _auto_lr(self, t):
        if t % 20 == 0 and t != 0:
            de = self.errors[-1] - self.errors[-2]
            if de > 0:
                self.lr *= 1.05
            else:
                self.lr *= 0.75

    def activation(self, x):
        """激活函数：sigmoid"""
        return 1 / (1 + np.exp((-1) * x))


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    # iris = load_iris()
    # sep = int(len(iris.data) * 0.7)
    # X_train, y_train = iris.data[:sep], iris.target[:sep]
    # X_test, y_test = iris.data[sep:], iris.target[sep:]
    X_train = np.array(
        [[60, 18.4], [85.5, 16.8], [64.8, 21.6], [61.5, 20.8], [87, 23.6], [82.8, 22.4], [69, 20], [93, 20.8], [51, 22],
         [75, 19.6], [64.8, 17.2],
         [43.2, 20.4], [84, 17.6], [49.2, 17.6], [47.4, 16.4], [33, 18.8], [51, 14], [63, 14.8]])
    y_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 对应训练数据的18个分类标
    X_test = np.array([[110.1, 19.2], [108, 17.6], [81, 20], [52.8, 20.8], [59.4, 16], [66, 18.4]])  # 6个测试数据
    y_test = np.array([1, 1, 1, 0, 0, 0])  # 对应6个测试数据的已知标

    xrange = range(2, 6, 1)
    acc = []
    error = []
    for k in xrange:
        knn = KNN(k).fit(X_train, y_train)
        # acc.append(knn.predict_accuracy(X_test, y_test))
        error.append(knn.predict_error_rate(X_test, y_test))

    # print(acc)
    print(error)
    # 动态图
    fig, ax = plt.subplots()
    x1, y1 = [], []
    for i, x in enumerate(xrange):
        x1.append(x)
        y1.append(error[i])
        ax.cla()
        ax.plot(x1, y1)
    plt.show()
