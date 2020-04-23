import numpy as np
import pandas as pd
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
            indexs = np.argsort(dicts)    # 根据值得索引排序
            cnts = np.zeros(self.classnum)
            for idx in indexs[:self.k]:             # 统计最近k个数据的标签
                tag = self.y_train[idx]
                cnts[tag] += 1
            imax = np.argmax(cnts)
            pre = self.classes[imax]                # 找到周围出现最多的标签
            # print(pre)
            y_pre[ith] = pre                        # 记录每一个数据对应预测的分类结果
        return y_pre

            
    def predict_accuracy(self, X_test, y_test):
        y_pre = self.predict(X_test)                # 正确率
        return (y_pre == y_test).sum() * 1.0 / len(X_test)
    
    def predict_error_rate(self, X_test, y_test):   # 错分率
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
        self.matrix = np.ones((self.nCategory, self.nFeature))      # 使用拉普拉斯平滑lambda=1
        self._nWordCate = np.zeros(self.nCategory) + self.nCategory # K * lambda

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

        for r in range(len(Xtest)):     # 每一行数据
            pCates = np.zeros(self.nCategory)
            # 遍历各个分类求得对于每种分类的后验概率
            for i in range(self.nCategory):     # 每一种分类
                pCates[i] = np.sum(self.logMatrix[i] * Xtest[r]) + np.log(self.pPriors[i])
            y_pred[r] = self.category[np.argmax(pCates)]
        return y_pred




if __name__ == '__main__':
    from sklearn.datasets import load_iris
    # iris = load_iris()
    # sep = int(len(iris.data) * 0.7)
    # X_train, y_train = iris.data[:sep], iris.target[:sep]
    # X_test, y_test = iris.data[sep:], iris.target[sep:]
    X_train=np.array([[60,18.4],[85.5,16.8],[64.8,21.6],[61.5,20.8],[87,23.6],[82.8,22.4],[69,20],[93,20.8],[51,22],[75,19.6],[64.8,17.2],
       [43.2,20.4],[84,17.6],[49.2,17.6],[47.4,16.4],[33,18.8],[51,14],[63,14.8]])
    y_train=np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])#对应训练数据的18个分类标
    X_test=np.array([[110.1,19.2],[108,17.6],[81,20],[52.8,20.8],[59.4,16],[66,18.4]])#6个测试数据
    y_test=np.array([1,1,1,0,0,0])#对应6个测试数据的已知标

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
        

        
    


        

   