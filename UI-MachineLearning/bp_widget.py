import tkinter as tk
from tkinter.font import Font
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_algorithm import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import tkinter.filedialog as filedialog
from data_core import DataSet
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class BPWidget(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self._data = DataSet()
        self._image_size = (400, 400)  # 高， 宽
        self._sample_size = (32, 32)  # 高， 宽
        self._font = Font(family='microsoft yahei', size=12)
        self._valImage = None
        self._image = tk.PhotoImage(file='image/logo1.png')  # 初始化图片
        self._new_image = tk.PhotoImage(file='image/logo1.png')  # 初始化图片
        self._tag_idx = tk.IntVar()
        # self._tags = [chr(x) for x in range(ord('A'), ord('Z') + 1)]
        self._tags = [x for x in range(10)]
        self._net = None
        self._pred = None

        # Modules
        self._img_module(master)
        self._btn_module(master)
        self._graph_module(master)
        self._stat_module(master)

        # 初始化部分val
        self._update_val(component='Tag')

    def _img_module(self, master):
        self._frm_img = tk.Frame(master)

        # 识别结果
        frm_res = tk.Frame(self._frm_img)
        tk.Label(frm_res, text='识别结果', font=self._font).grid(row=0, column=0)
        self._res = tk.Entry(frm_res, justify='center')
        self._res.grid(row=0, column=1)
        frm_res.pack(pady=10)

        # 图片
        self._canvas = tk.Canvas(self._frm_img, width=self._image_size[1], height=self._image_size[0])
        self._canvas.create_image(0, 0, anchor='nw', image=self._image)
        self._canvas.pack()
        self._frm_img.grid(row=0, column=0, padx=10)

        # In Btn Frame

    def _btn_module(self, master):
        self._frm_btn = tk.Frame(master)

        # 滑动条
        frm_tag = tk.Frame(self._frm_btn, pady=20)
        self._tag_lbl = tk.Entry(frm_tag, justify='center', width=10)
        self._tag_lbl.grid(row=0, column=0)
        slider = tk.Scale(frm_tag, from_=0, to=len(self._tags) - 1, orient=tk.HORIZONTAL,
                          length=300, variable=self._tag_idx,
                          command=lambda x: self._update_val(component='Tag')).grid(row=0, column=1)  # TODO: update
        frm_tag.pack()

        # 功能按钮
        frm_btn = tk.Frame(self._frm_btn)
        btn_mess = ['选择图片', '添加样本', '保存样本',
                    '装入样本', '训练网络', '识别样本',
                    '保存网络', '装入网络', '删除样本']
        btn_func = [self._open_file, self._add_sample, self._save_sample,
                    self._load_sample, self._train_net, self._recognize,
                    None, None, None, ]
        btns = [tk.Button(frm_btn, text=btn_mess[idx], font=self._font, padx=10, command=btn_func[idx])
                for idx, _ in enumerate(btn_mess)]
        rows, cols = 3, 3
        for r in range(rows):
            for c in range(cols):
                btns[r * rows + c].grid(row=r, column=c, padx=20, pady=10)
        frm_btn.pack()
        self._frm_btn.grid(row=1, column=0)

    def _graph_module(self, master):
        self._frm_graph = tk.Frame(master)
        # 小标题
        tk.Label(self._frm_graph, text='训练误差曲线', font=self._font).pack(pady=10)

        # 曲线图
        f, ax = plt.subplots(figsize=(self._image_size[0] // 100, self._image_size[1] // 100))
        self._new_canvas = FigureCanvasTkAgg(f, master=self._frm_graph)
        self._new_canvas.draw()
        self._new_canvas.get_tk_widget().pack(side=tk.TOP,
                                              fill=tk.BOTH,
                                              expand=tk.YES)

        self._frm_graph.grid(row=0, column=1, padx=10)

    def _stat_module(self, master):
        self._frm_stat = tk.Frame(master)

        tk.Label(self._frm_stat, text='样本计数:', font=self._font).grid(row=0, column=0, pady=10)
        tk.Label(self._frm_stat, text='训练次数:', font=self._font).grid(row=1, column=0, pady=10)
        tk.Label(self._frm_stat, text='训练误差:', font=self._font).grid(row=2, column=0, pady=10)
        self._n_sample = tk.Entry(self._frm_stat, justify='center')
        self._epoch = tk.Entry(self._frm_stat, justify='center')
        self._error = tk.Entry(self._frm_stat, justify='center')
        self._n_sample.grid(row=0, column=1)
        self._epoch.grid(row=1, column=1)
        self._error.grid(row=2, column=1)

        self._frm_stat.grid(row=1, column=1)

    # TODO: need to update
    def _add_sample(self):
        """添加训练样本"""
        # 统一样本的尺寸
        tmp_image = cv2.resize(self._valImage, self._sample_size)

        # BGR转灰度, ravel成一维
        gray = cv2.cvtColor(tmp_image, cv2.COLOR_BGRA2GRAY).ravel()

        # 标签为数值
        tmp_tag = self._tag_idx.get()

        # 保存数据
        self._data.add_sample(data=gray, tag=tmp_tag)

        # 标签数量更新
        self._update_val(component='NSample')

        print('Add Sample %s Done.' % tmp_tag)

    def _open_file(self):
        """打开文件，导入图片"""
        filename = filedialog.askopenfilename(title='打开文件', initialdir='image/Digits/',
                                              filetypes=[('jpg文件', '*.jpg'), ('png文件', '*.png')])
        if len(filename) == 0:  # 如果直接关闭，返回''
            print('Open file Error.')
            return
        self._valImage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_COLOR)  # 读出的图片值
        # print('Image:', self._valImage.shape)  # 打印shape

        # 写暂时文件tmp-origin.png
        tmpFilename = 'image/tmp-origin.png'
        cv2.imwrite(tmpFilename, cv2.resize(self._valImage, self._image_size))

        # 显示原始图片
        self._image.config(file=tmpFilename)

    def _save_sample(self):
        """保存样本"""
        filename = filedialog.asksaveasfilename(title='保存样本', initialdir='files/sample/',
                                                filetype=[('pk文件', '*.pk')])
        if len(filename) == 0:
            return

        if not filename.endswith('.pk'):
            filename = filename + '.pk'

        with open(filename, 'wb') as f:
            pickle.dump(self._data, f)

        print('Save Sample Done.')

    def _load_sample(self):
        """加载样本"""
        filename = filedialog.askopenfilename(title='加载样本', initialdir='files/sample/',
                                              filetypes=[('pk文件', '*.pk')])
        if len(filename) == 0:
            return

        with open(filename, 'rb') as f:
            self._data = pickle.load(f)

        # 更新样本计数
        self._update_val(component='NSample')

        print('Load Sample Done.')

    def _train_net(self):
        """训练网络"""
        # 获取数据集data, target
        X = self._data.get_datas()
        y = self._data.get_tags()
        X_std = StandardScaler().fit_transform(X)

        # 设置网络
        if self._net is None:
            self._net = BP(n_input=X.shape[1], n_output=1,
                           epoch=50, accuary=0.000001, lr=0.01, is_auto_lr=True)

        # 训练
        self._net.fit(X_std, y)

        # 曲线图
        self._draw_graph()

        # 更新统计数据
        self._update_val(component='Epoch')
        self._update_val(component='Error')



    def _draw_graph(self):
        """画曲线图"""
        fig, ax = plt.subplots(figsize=(self._image_size[0]//100, self._image_size[1]//100))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')
        # ax.set_autoscale_on(True)
        y = self._net.errors
        xrange = range(len(y))
        ax.plot(xrange, y)

        # 显示在界面上
        self._new_canvas.figure = fig
        self._new_canvas.draw()

    def _recognize(self):
        """识别样本"""
        filename = filedialog.askopenfilename(title='打开文件', initialdir='image/Digits/',
                                              filetypes=[('jpg文件', '*.jpg'), ('png文件', '*.png')])
        if len(filename) == 0:  # 如果直接关闭，返回''
            print('Open file Error.')
            return
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_COLOR)  # 读出的图片值

        # 统一样本的尺寸
        tmp_image = cv2.resize(img, self._sample_size)

        # BGR转灰度,再转成shape=(1, -1)
        gray = cv2.cvtColor(tmp_image, cv2.COLOR_BGRA2GRAY).reshape((1, -1))

        # 得到结果
        y_prob = self._net.predict(gray)
        self._pred = self._tags[np.argmax(y_prob)]
        print('y_prob', y_prob.ravel())

        # 更新结果
        self._update_val(component='Result')

        print('Recognize Done.')

    def _update_val(self, component=None):
        """
        更新组件数据
        :param components: 标签选择 -- Tag
                            识别结果 -- Result
                            样本计数 -- NSample
                            训练次数 -- Epoch
                            训练误差 -- Error
        :return:
        """
        # TODO: 补充内容
        if component == 'Tag':
            self._tag_lbl.delete(0, tk.END)
            self._tag_lbl.insert(0, self._tags[self._tag_idx.get()])
        elif component == 'NSample':
            self._n_sample.delete(0, tk.END)
            self._n_sample.insert(0, len(self._data))
        elif component == 'Epoch':
            self._epoch.delete(0, tk.END)
            self._epoch.insert(0, len(self._net.errors))
        elif component == 'Error':
            self._error.delete(0, tk.END)
            self._error.insert(0, self._net.errors[-1])
        elif component == 'Result':
            self._res.delete(0, tk.END)
            self._res.insert(0, self._pred)
