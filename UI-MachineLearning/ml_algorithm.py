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
        y_pre = []
        for item in X_test:
            dicts = []
            for d in self.X_train:
                dicts.append(np.linalg.norm(item - d))  # 欧式距离
            indexs = np.argsort(np.array(dicts))    # 根据值得索引排序
            cnts = np.zeros(self.classnum)
            for idx in indexs[:self.k]:             # 统计最近k个数据的标签
                tag = self.y_train[idx]
                cnts[tag] += 1
            imax = np.argmax(cnts)
            pre = self.classes[imax]                # 找到周围出现最多的标签
            # print(pre)
            y_pre.append(pre)
        return np.array(y_pre)

            
    def predict_accuracy(self, X_test, y_test):
        y_pre = self.predict(X_test)                # 正确率
        return (y_pre == y_test).sum() * 1.0 / len(X_test)
    
    def predict_error_rate(self, X_test, y_test):   # 错分率
        return 1 - self.predict_accuracy(X_test, y_test)
    


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

    xrange = range(5, 6, 1)
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
        plt.pause(0.1)
        

        
    


        

   