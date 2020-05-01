"""This file is to implement operator like
    Prewitt, Sobel
"""

import numpy as np
import math

_L_DEFAULT = 256

def prewitt(X, L=_L_DEFAULT):
    """
    dz/dx --> [[-1 -1 -1],
                [0  0  0],
                [1  1  1]],
    dz/dy --> [[-1  0  1],
                [-1 0  1],
                [-1 0  1]]
    :param X:
    :return: updated X
    """
    X = _normalize(X, L)
    newGray = X.copy()
    rows, cols = len(X), len(X[0])
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            dx = -1 * X[i - 1, j - 1] - X[i - 1, j] - X[i - 1, j + 1] + X[i + 1, j - 1] + X[i + 1, j] + X[i + 1, j + 1]
            dy = -1 * X[i-1, j-1] - X[i, j-1] - X[i+1, j-1] + X[i-1, j+1] + X[i, j+1] + X[i+1, j+1]
            newGray[i, j] = int(abs(dx) + abs(dy))      # 用|dx|+|dy| 代替 (dx^2+dy^2)^0.5 减少计算量
    newGray *= (L-1)
    return newGray


def sobel(X, L=_L_DEFAULT):
    """
    dz/dx --> [[-1 -2 -1],
                [0  0  0],
                [1  2  1]],
    dz/dy --> [[-1  0  1],
                [-2 0  2],
                [-1 0  1]]
    :param X:
    :return: updated X
    """
    X = _normalize(X, L)
    newGray = X.copy()
    newGray /= float(L-1)
    rows, cols = len(X), len(X[0])
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            dx = -1 * X[i - 1, j - 1] - 2*X[i - 1, j] - X[i - 1, j + 1] + X[i + 1, j - 1] + 2*X[i + 1, j] + X[i + 1, j + 1]
            dy = -1 * X[i-1, j-1] - 2*X[i, j-1] - X[i+1, j-1] + X[i-1, j+1] + 2*X[i, j+1] + X[i+1, j+1]
            newGray[i, j] = int(abs(dx) + abs(dy))      # 用|dx|+|dy| 代替 (dx^2+dy^2)^0.5 减少计算量
    newGray *= (L-1)
    return newGray


def laplace(X, L=_L_DEFAULT):
    """
    dz/dx --> [[-1  -1 -1],
                [-1  8 -1],
                [-1 -1 -1]],
    dz/dy --> [[-1  -1 -1],
                [-1  8 -1],
                [-1 -1 -1]]
    :param X:
    :return: updated X
    """
    X = _normalize(X, L)
    newGray = X.copy()
    newGray /= (L-1)
    rows, cols = len(X), len(X[0])
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            dx = -1*X[i-1,j-1] - X[i-1,j] - X[i-1,j+1] \
                    - X[i,j-1] + 8*X[i,j] - X[i,j+1] \
                    - X[i+1,j-1] - X[i+1,j] - X[i+1,j+1]
            dy = -1*X[i-1,j-1] - X[i-1,j] - X[i-1,j+1] \
                 - X[i,j-1] + 8*X[i,j] - X[i,j+1] \
                 - X[i+1,j-1] - X[i+1,j] - X[i+1,j+1]
            newGray[i, j] = int(abs(dx) + abs(dy))      # 用|dx|+|dy| 代替 (dx^2+dy^2)^0.5 减少计算量
    newGray *= (L-1)
    return newGray

def color2gray(X):
    """
    彩色图片转成灰度值
    Y = 0.299*R + 0.587*G + 0.114*B
    :param X: 三维的图片
    :return: 转换成的灰度值
    """
    if len(X.shape) == 3:
        tmp = np.array([0.114, 0.587, 0.299])       # 实际图片存储按BGR顺序存储
        return np.floor(np.dot(X, tmp) + 0.5).astype(np.uint8)      # +0.5向下取整，注意float-->uint8
    elif len(X.shape) == 2:
        return X
    else:
        raise Exception("Can't transfer color to gray.")


def gray2bin(X, threshold=_L_DEFAULT // 2):
    """灰度值二值化"""
    Y = X.copy()
    Y[Y < threshold] = 0
    Y[Y >= threshold] = 1
    return Y

def bin2gray(X, L=_L_DEFAULT):
    """二值图转成灰度值"""
    Y = X * (L-1)
    return Y


def get_hist(X, L=_L_DEFAULT):
    """获取直方图列表"""
    if len(X.shape) > 2:
        raise Exception("Can't get hists from more than 2D (not gray image)")

    grays = np.zeros(L)
    rows, cols = X.shape
    for r in range(rows):
        for c in range(cols):
            grays[X[r, c]] += 1
    return grays / (rows * cols)


def hist_equal(X, L=_L_DEFAULT):
    """直方图均衡化"""
    # 获取直方图数组
    pgrays = get_hist(X, L)

    # 构建新直方图数组 -- 取整扩展法
    newPGrays = np.zeros_like(pgrays)
    for i, s in enumerate(np.cumsum(pgrays)):
        newPGrays[i] = int(math.floor((L-1) * s + 0.5))

    # 灰度值均衡化
    Y = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i, j] = newPGrays[X[i, j]]
    return Y


def _normalize(X, L=_L_DEFAULT):
    """归一化"""
    Y = X.copy()
    Y = Y / L
    return Y


def bimode_cut(X, threshold_type='mean', L=_L_DEFAULT):
    """
    双峰法阈值分割
    :param X:
    :param threshold_type: 'mean' -- 双峰的平均值
                        'low' -- 双峰间最低点
    :return: 二值矩阵
    """
    # 生成直方图
    Y = X.copy()
    hist = get_hist(Y, L)

    for t in range(1000):
        # 平滑处理
        hist[0] = (hist[0] + hist[0] + hist[1]) / 3
        for i in range(1, len(hist)-1):
            hist[i] = (hist[i-1] + hist[i] + hist[i+1]) / 3
        hist[-1] = (hist[-1] + hist[-1] + hist[-2]) / 3

        # 判断是否双峰图像，找出阈值
        threshold = None
        if not _is_dimode(hist):        # 不是双峰
            print('smoothing', t)
            continue
        else:                           # 双峰
            if threshold_type == 'mean':
                peak1, peak2 = None, None
                for i in range(1, len(hist)):
                    if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                        if not peak1:
                            peak1 = i
                        else:
                            peak2 = i
                            threshold = (peak1 + peak2) // 2
                            break
            elif threshold_type == 'low':
                peakFound = False
                for i in range(1, len(hist)):
                    if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                        peakFound = True
                    if peakFound and hist[i] <= hist[i-1] and hist[i] <= hist[i+1]:
                        threshold = i
                        break
            else:
                raise Exception('No such threshold type:', threshold_type)

            # 二值化
            if threshold:
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Y[i,j] = L-1 if X[i,j] >= threshold else 0
                # 退出外层1000循环
                break
    return Y



def _is_dimode(hist):
    """判断直方图是否双峰"""
    nPeak = 0
    for i in range(1, len(hist)-1):
        if hist[i-1] < hist[i] and hist[i] > hist[i+1]:
            nPeak += 1
            if nPeak > 2:
                break
    return True if nPeak == 2 else False



def dilation(X, kernel_size=3):
    """简单膨胀操作"""
    return _dilation_or_frosion(X, way='Dilation', kernel_size=kernel_size)

def frosion(X, kernel_size=3):
    """简单腐蚀操作"""
    return _dilation_or_frosion(X, way='Frosion', kernel_size=kernel_size)


def _dilation_or_frosion(X, way=None, kernel_size=3):
    """简单的膨胀或者腐蚀操作，没有自定义模板功能"""
    kernel_size = int(kernel_size)
    Y = X.copy()
    if way == None:
        return Y

    if X.max() != 1 or X.min() != 0:
        raise ValueError('Image must be two-valued.')

    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError('Image must be odd or kernel size must bigger than 1.')

    gap = kernel_size // 2
    for i in range(gap, X.shape[0]-gap+1):
        for j in range(gap, X.shape[1]-gap+1):
            tmp = X[i-gap:i+gap+1, j-gap:j+gap+1]
            if way == 'Dilation':
                val = np.max(tmp)
            elif way == 'Frosion':
                val = np.min(tmp)
            else:
                val = X[i,j]        # TODO: 可以补充模板
            Y[i,j] = val
    return Y

def opening(X):
    """
    开操作， opening(A, B) = dilation(frosion(A, B), B)
    """
    return dilation(frosion(X))


def closing(X):
    """
    闭操作，closing(A, B) = frosion(dilation(A, B), B)
    """
    return frosion(dilation(X))


def edge_extraction(X):
    """
    边界提取， beta(A) = A - frosion(A, B)
    """
    return X - frosion(X)