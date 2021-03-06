"""
This Python module is to finish some units like "Frame" and "Menu".
"""

import tkinter as tk
import tkinter.filedialog as filedialog
import myoperator
import numpy as np
import matplotlib.pyplot as plt
import cv2

'''主界面'''


class MainWidget(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self._imageSize = (400, 400)
        self._valImage = None
        self._valCompare = None
        self._valUpdate = None
        self._nowOperator = None  # 当前使用的算子为None，即原图的灰度值
        self._valS = tk.IntVar()
        self._image = tk.PhotoImage(file='image/temp2.png')  # 初始化图片
        self._newImage = tk.PhotoImage(file='image/temp2.png')  # 初始化图片
        self._frmL = tk.Frame(master, padx=10, pady=10)
        self._frmR = tk.Frame(master, padx=10, pady=10)
        self._left_part()
        self._right_part()
        self._frmL.pack(side='left')
        self._frmR.pack(side='right')

    def _left_part(self):
        # 小标题
        tk.Label(self._frmL, text='小标题').pack()

        # 图片,画布
        self._canvas = tk.Canvas(self._frmL, width=self._imageSize[0], height=self._imageSize[1])
        self._canvas.create_image(0, 0, anchor='nw', image=self._image)
        self._canvas.pack()

        # 标记图片
        self._frmRBtn = tk.Frame(self._frmL)
        self._varLabel = ''
        rbtns = [tk.Radiobutton(self._frmRBtn, text=str(idx), variable=self._varLabel, value=str(idx))
                 for idx in range(10)]
        for idx, rbtn in enumerate(rbtns):
            rbtn.grid(row=0, column=idx)
        self._frmRBtn.pack()

        # 功能按键
        self._frmBtn = tk.Frame(self._frmL)
        btnMess = ['选择图片', '对比图片', '备用按钮', '备用按钮', '备用按钮', '备用按钮', '备用按钮', '备用按钮', '备用按钮']
        btnFunc = [self.open_file, self.open_compare, None,
                   None, None, None,
                   None, None, None]  # 功能方法名
        btns = [tk.Button(self._frmBtn, text=btnMess[idx], padx=10, command=btnFunc[idx])
                for idx, _ in enumerate(btnMess)]
        rows, cols = 3, 3
        for r in range(rows):
            for c in range(cols):
                btns[r * rows + c].grid(row=r, column=c, padx=20, pady=10)
        self._frmBtn.pack()

    def _right_part(self):
        # 小标题
        self._subTitle = tk.Label(self._frmR, text='小标题')
        self._subTitle.pack()

        # 生成的图片
        self._newCanvas = tk.Canvas(self._frmR, width=self._imageSize[0], height=self._imageSize[1], bg='white')
        self._newCanvas.create_image(0, 0, anchor='nw', image=self._newImage)
        self._newCanvas.pack()

        # 滑动条
        self._frmTh = tk.Frame(self._frmR, pady=50)
        tk.Label(self._frmTh, text='设定阈值', font=('Consolas', 14)).grid(row=0, column=0)
        self._slider = tk.Scale(self._frmTh, from_=1, to=100, orient=tk.HORIZONTAL,
                                length=300, variable=self._valS, command=None)
        self._slider.grid(row=0, column=1)
        tk.Label(self._frmTh, text='边缘检测与二值图像阈值比例因子').grid(row=1, column=1)
        self._frmTh.pack()

    def open_file(self):
        """打开文件，导入图片"""
        filename = filedialog.askopenfilename(title='打开文件', initialdir='image',
                                              filetypes=[('jpg文件', '*.jpg'), ('png文件', '*.png')])
        if len(filename) == 0:  # 如果直接关闭，返回''
            print('Open file Error.')
            return
        self._valImage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_COLOR)  # 读出的图片值
        print('Image:', self._valImage.shape)  # 打印shape

        # 写暂时文件tmp-origin.png
        tmpFilename = 'image/tmp-origin.png'
        cv2.imwrite(tmpFilename, cv2.resize(self._valImage, self._imageSize))

        # 显示原始图片
        self._image.config(file=tmpFilename)

    def open_compare(self):
        """打开对比文件，导入图片"""
        filename = filedialog.askopenfilename(title='打开对比文件', initialdir='image',
                                              filetypes=[('jpg文件', '*.jpg'), ('png文件', '*.png')])
        if len(filename) == 0:  # 如果直接关闭，返回''
            print('Open file Error.')
            return
        self._valCompare = cv2.imread(filename, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_COLOR)  # 读出的图片值
        print('Compare Image:', self._valCompare.shape)  # 打印shape

        # 写暂时文件tmp-compare.png
        tmpFilename = 'image/tmp-compare.png'
        cv2.imwrite(tmpFilename, cv2.resize(self._valCompare, self._imageSize))

    def update_image(self, _operator=None, _thVal=None):
        """
        转换原图片，并且显示新的图片
        :param _operator: 操作的算子，默认是None --> 转换为灰度值；
                                        Prewitt, Sobel, Laplace --> 各种算子处理
                                        TODO: 其他补充的算子
        :param _thVal: 阈值，两值化分割
        :return:
        """
        self._nowOperator = _operator  # 保存当前使用的算子
        # X = (np.array(cv2.cvtColor(self._valImage, cv2.COLOR_BGRA2GRAY)))  # 转换成灰度值
        X = myoperator.color2gray(self._valImage)  # 转换成灰度值

        if _operator == 'Prewitt':  # Prewitt算子
            self._valUpdate = myoperator.prewitt(X)
        elif _operator == 'Sobel':  # Sobel算子
            self._valUpdate = myoperator.sobel(X)
        elif _operator == 'Laplace':  # Laplace算子
            self._valUpdate = myoperator.laplace(X)
        elif _operator == 'Gray':  # 转成灰度值
            self._valUpdate = X
        elif _operator == 'HistEqual':  # 直方图均衡化
            self._valUpdate = myoperator.hist_equal(X)
        elif _operator == 'Gray2Bin':  # 二值化
            X2 = myoperator.gray2bin(X)
            self._valUpdate = myoperator.bin2gray(X2)
        elif _operator == 'BimodeMean':  # 双峰法平均
            self._valUpdate = myoperator.bimode_cut(X, threshold_type='mean')
        elif _operator == 'BimodeLow':  # 双峰法最小
            self._valUpdate = myoperator.bimode_cut(X, threshold_type='low')
        elif _operator == 'Dilation':  # 膨胀
            X2 = myoperator.gray2bin(X)
            print('Gray to Bin --> Bin to Gray')
            self._valUpdate = myoperator.bin2gray(myoperator.dilation(X2))
        elif _operator == 'Frosion':  # 腐蚀
            X2 = myoperator.gray2bin(X)
            print('Gray to Bin --> Bin to Gray')
            self._valUpdate = myoperator.bin2gray(myoperator.frosion(X2))
        elif _operator == 'Opening':  # 开操作
            X2 = myoperator.gray2bin(X)
            print('Gray to Bin --> Bin to Gray')
            self._valUpdate = myoperator.bin2gray(myoperator.opening(X2))
        elif _operator == 'Closing':  # 闭操作
            X2 = myoperator.gray2bin(X)
            print('Gray to Bin --> Bin to Gray')
            self._valUpdate = myoperator.bin2gray(myoperator.closing(X2))
        elif _operator == 'EdgeExtraction':  # 边界提取
            X2 = myoperator.gray2bin(X)
            print('Gray to Bin --> Bin to Gray')
            self._valUpdate = myoperator.bin2gray(myoperator.edge_extraction(X2))
        elif _operator == 'ObjectSpot':  # 对象识别
            if self._valCompare is None:
                raise Exception('Please select compare image.')
            X_spot = myoperator.color2gray(self._valCompare)
            loss, (xx, yy), side = myoperator.object_spot(X, X_spot, incr=10)
            self._valUpdate = self._valImage
            self._valUpdate[xx:xx+side, yy:yy+side] = myoperator.reverse(self._valUpdate[xx:xx+side, yy:yy+side])
            print('loss:', loss)
        elif _operator == 'SSDA':       # SSDA序贯相似度检测
            if self._valCompare is None:
                raise Exception('Please select compare image.')
            X_spot = myoperator.color2gray(self._valCompare)
            max_count, xx, yy = myoperator.ssda(X, X_spot, threshold=1)
            self._valUpdate = self._valImage
            xside, yside = X_spot.shape
            self._valUpdate[xx:xx+xside, yy:yy+yside] = myoperator.reverse(self._valUpdate[xx:xx+xside, yy:yy+yside])
            print('max_count:', max_count)

        else:  # 不转换 或者 使用没定义算子
            self._valUpdate = X

        # 如果考虑滑动条阈值, 二值图像
        if _thVal:  # 阈值
            s = 0.1 * _thVal
            gradientAvg = np.mean(s * self._valUpdate)  # 梯度均值
            self._valUpdate[self._valUpdate < gradientAvg] = 1
            self._valUpdate[self._valUpdate >= gradientAvg] = 0

        # 写入磁盘和显示图片
        self._write_new_image(self._valUpdate)

    def _write_new_image(self, X, filename='image/tmp-update.png'):

        # 写到磁盘中
        cv2.imwrite(filename, cv2.resize(X, self._imageSize))

        # 展示 -- 修改变量self._newImage
        self._newImage.config(file=filename)

    # todo: 滑动条连续触发，需要换另一种方法使用滑动条
    def _slider_update(self, thVal):
        """通过滑动条控制阈值分为黑白两色"""
        thVal = int(thVal)
        self.update_image(_operator=self._nowOperator, _thVal=thVal)

    def show_hist(self, filename='image/tmp-hist.png'):
        """生成原图像的直方图图片，写入磁盘，显示图片"""
        # 获得灰度值
        X = myoperator.color2gray(self._valImage)

        # 生成直方图
        hist = myoperator.get_hist(X)

        # 展示图片
        imageSize = (self._imageSize[0] // 100, self._imageSize[1] // 100)
        fig, axe = plt.subplots(figsize=imageSize)  # 默认返回一个子图
        axe.plot(range(len(hist)), hist)
        axe.set_xlabel('Gray value')
        axe.set_ylabel('Proportion')
        fig.savefig(filename, dpi=100)
        self._newImage.config(file=filename)

    def show_humoments(self, filename='image/tmp-humoments.png'):
        """生成原图像的Hu矩直方图图片，写入磁盘，显示图片"""

        # cv2的HuMoments
        # log_ims = np.log(np.abs(cv2.HuMoments(cv2.moments(myoperator.color2gray(self._valImage))))).ravel()
        # print('log-cv2.HuMoments:', log_ims)

        # 生成不变矩列表
        X = myoperator.color2gray(self._valImage)
        log_ims = np.log(np.abs(myoperator.humoments(X)))  # 先取绝对值，后取对数（不变矩可能为负数）
        print('log-HuMoments:', log_ims)

        # 展示图片
        imageSize = (self._imageSize[0] // 100, self._imageSize[1] // 100)
        fig, axe = plt.subplots(figsize=imageSize)  # 默认返回一个子图
        xrange = range(1, len(log_ims) + 1)
        axe.bar(xrange, log_ims, tick_label=xrange)
        axe.set_xlabel('Invariant Moments')
        axe.set_ylabel('log-val')
        axe.set_title('Log - HuMoments')
        fig.savefig(filename, dpi=100)
        self._newImage.config(file=filename)







