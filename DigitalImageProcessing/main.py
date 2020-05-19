"""
This python file is to build "MyWindow" class using Tk.
"""

import tkinter as tk
from units import MainWidget


class MyWindow():

    def __init__(self):
        self.root = tk.Tk()
        self.widget = MainWidget(self.root)
        self.menu = tk.Menu(self.root)
        # todo: menu function
        self._init_menu()
        self.widget.pack()
        self.root.config(menu=self.menu)
        self.root.mainloop()

    def _init_menu(self):
        """绑定菜单和主界面的功能"""
        # 声明菜单变量
        menuFile = tk.Menu(self.menu, tearoff=True)
        menuHist = tk.Menu(self.menu, tearoff=True)
        menuEdge = tk.Menu(self.menu, tearoff=True)
        menuCut = tk.Menu(self.menu, tearoff=True)
        menuBin = tk.Menu(self.menu, tearoff=True)
        menuExpr = tk.Menu(self.menu, tearoff=True)
        menuSpot = tk.Menu(self.menu, tearoff=True)

        # 绑定菜单名字
        self.menu.add_cascade(label='图像文件操作', menu=menuFile)
        self.menu.add_cascade(label='图像直方图处理', menu=menuHist)
        self.menu.add_cascade(label='图像边缘检测', menu=menuEdge)
        self.menu.add_cascade(label='图像阈值分割', menu=menuCut)
        self.menu.add_cascade(label='图像形态学', menu=menuBin)
        self.menu.add_cascade(label='图像表示描述', menu=menuExpr)
        self.menu.add_cascade(label='图像对象识别', menu=menuSpot)

        # 添加菜单内容
        # 文件操作
        menuFile.add_command(label='新建', command=None)
        menuFile.add_command(label='打开', command=None)
        menuFile.add_command(label='保存', command=None)
        menuFile.add_command(label='另存为', command=None)
        # 直方图
        menuHist.add_command(label='彩色转灰度图像', command=lambda: self._update('Gray'))
        menuHist.add_command(label='图像二值化处理', command=lambda: self._update('Gray2Bin'))
        menuHist.add_command(label='计算直方图', command=self._show_hist)
        menuHist.add_command(label='直方图均衡化', command=lambda: self._update('HistEqual'))
        # 边缘检测
        menuEdge.add_command(label='Sobel算子', command=lambda: self._update('Sobel'))
        menuEdge.add_command(label='Prewitt算子', command=lambda: self._update('Prewitt'))
        menuEdge.add_command(label='Canny算子', command=lambda: self._update(None))
        menuEdge.add_command(label='Laplace算子', command=lambda: self._update('Laplace'))
        # 阈值分割
        menuCut.add_command(label='双峰法平均阈值分割', command=lambda: self._update('BimodeMean'))
        menuCut.add_command(label='双峰法最小阈值分割', command=lambda: self._update('BimodeLow'))
        menuCut.add_command(label='局部阈值分割', command=None)
        menuCut.add_command(label='裂合4叉树分割', command=None)
        menuCut.add_command(label='区域扩展分割', command=None)
        # 形态学（二值）
        menuBin.add_command(label='膨胀', command=lambda: self._update('Dilation'))
        menuBin.add_command(label='腐蚀', command=lambda: self._update('Frosion'))
        menuBin.add_command(label='开操作', command=lambda: self._update('Opening'))
        menuBin.add_command(label='闭操作', command=lambda: self._update('Closing'))
        menuBin.add_command(label='击中变换', command=None)
        menuBin.add_command(label='边界提取', command=lambda: self._update('EdgeExtraction'))
        # 表示描述
        menuExpr.add_command(label='区域hu矩', command=self._show_humoments)
        # 图像识别
        menuSpot.add_command(label='对象识别', command=lambda: self._update('ObjectSpot'))

    def _update(self, _operator):
        """统一更新方法"""
        self.widget.update_image(_operator=_operator)
        print('Update Done -- %s.' % _operator)

    def _show_hist(self):
        """展示直方图"""
        self.widget.show_hist()

    def _show_humoments(self):
        """展示Hu矩直方图"""
        self.widget.show_humoments()


if __name__ == '__main__':
    window = MyWindow()
