import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_algorithm import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class MainWidget(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        # left frame
        self.frm_l = tk.Frame(master)
        # right frame
        self.frm_r = tk.Frame(master)

        '''
        left frame ( Canvas, RadioButtons, Buttons )
        '''
        # Canvas
        self.canvas = tk.Canvas(self.frm_l, height=300, width=300, bg='gray')
        # filename = tk.PhotoImage(file='machinelearning.gif')
        # image = canvas.create_image(0, 0, anchor='nw', image=filename)
        self.canvas.pack()
        # RadioButton
        frm_rbs = tk.Frame(self.frm_l)
        var = ''
        rbs = [tk.Radiobutton(frm_rbs, text=str(idx), variable=var, value=str(idx))
               for idx in range(10)]
        for idx, rb in enumerate(rbs):
            rb.grid(row=0, column=idx)
        frm_rbs.pack(pady=10)
        # Buttons
        frm_btns = tk.Frame(self.frm_l, )
        btns_mess = ['装载文件', '添加样本', '训练网络', '识别样本', '保存样本', '装入网络']
        rows, cols = 2, 3
        countid = 0
        btns = [tk.Button(frm_btns, text=m, padx=10) for m in btns_mess]
        for r in range(rows):
            for c in range(cols):
                btns[countid].grid(row=r, column=c, padx=20, pady=10)
                countid += 1
        frm_btns.pack()

        '''
        right frame (canvas, Entry, btns)
        '''
        # Canvas
        # self.convas_r = tk.Canvas(self.frm_r, height=300, width=300, bg='gray')
        f, ax = plt.subplots()
        self.canvas_r = FigureCanvasTkAgg(f, master=self.frm_r)
        self.canvas_r.draw()
        self.canvas_r.get_tk_widget().pack(side=tk.TOP,
                                           fill=tk.BOTH,
                                           expand=tk.YES)


        # Entry
        frm_in = tk.Frame(self.frm_r)
        input1 = tk.Entry(frm_in).pack()
        input2 = tk.Entry(frm_in).pack()
        frm_in.pack(pady=20)

        # model_btns
        self.frm_modelbtns = tk.Frame(self.frm_r)
        btn_knn = tk.Button(self.frm_modelbtns, text='KNN', padx=10, command=self.call_knn)
        btn_knn.grid(row=0, column=0, padx=20, pady=10)
        self.frm_modelbtns.pack()


        # frm_l, frm_r pack
        self.frm_l.pack(side='left')
        self.frm_r.pack(side='right')

    def call_knn(self):
        # iris = load_iris()
        # X_train, X_test, y_train , y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)
        X_train = np.array(
            [[60, 18.4], [85.5, 16.8], [64.8, 21.6], [61.5, 20.8], [87, 23.6], [82.8, 22.4], [69, 20], [93, 20.8],
             [51, 22], [75, 19.6], [64.8, 17.2],
             [43.2, 20.4], [84, 17.6], [49.2, 17.6], [47.4, 16.4], [33, 18.8], [51, 14], [63, 14.8]])
        y_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 对应训练数据的18个分类标
        X_test = np.array([[110.1, 19.2], [108, 17.6], [81, 20], [52.8, 20.8], [59.4, 16], [66, 18.4]])  # 6个测试数据
        y_test = np.array([1, 1, 1, 0, 0, 0])  # 对应6个测试数据的已知标

        xrange = range(1, 20, 2)
        acc = []
        error = []
        for k in xrange:
            knn = KNN(k).fit(X_train, y_train)
            # acc.append(knn.predict_accuracy(X_test, y_test))
            error.append(knn.predict_error_rate(X_test, y_test))

        # 动态图
        fig, ax = plt.subplots()
        ax.set_xlabel('K-points')
        ax.set_ylabel('Err-value')
        # ax.set_autoscale_on(True)
        x1, y1 = [], []
        for i, x in enumerate(xrange):
            x1.append(x)
            y1.append(error[i])
            ax.plot(x1, y1)

            # 显示在界面上
            self.canvas_r.figure=fig
            self.canvas_r.draw()
            


class MainMenu(tk.Menu):
    """
    menubar 文件操作， 图像处理， 机器学习
    """

    def __init__(self, master=None):
        tk.Menu.__init__(self, master)
        filemenu = tk.Menu(self, tearoff=False)
        imagemenu = tk.Menu(self, tearoff=False)
        mlmenu = tk.Menu(self, tearoff=False)

        self.add_cascade(label='文件操作', menu=filemenu)
        self.add_cascade(label='图像处理', menu=imagemenu)
        self.add_cascade(label='机器学习', menu=mlmenu)

        # file menu
        filemenu.add_command(label='新建', command=None)
        filemenu.add_command(label='打开', command=None)

        # machine learning menu




if __name__ == '__main__':
    window = tk.Tk()
    root = MainWidget().pack()
    menu = MainMenu()
    window.config(menu=menu)
    window.mainloop()
