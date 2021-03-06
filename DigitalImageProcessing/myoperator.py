"""This file is to implement operator like
    Prewitt, Sobel
"""

import numpy as np
import math
import cv2

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
            dy = -1 * X[i - 1, j - 1] - X[i, j - 1] - X[i + 1, j - 1] + X[i - 1, j + 1] + X[i, j + 1] + X[i + 1, j + 1]
            newGray[i, j] = int(abs(dx) + abs(dy))  # 用|dx|+|dy| 代替 (dx^2+dy^2)^0.5 减少计算量
    newGray *= (L - 1)
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
    newGray /= float(L - 1)
    rows, cols = len(X), len(X[0])
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            dx = -1 * X[i - 1, j - 1] - 2 * X[i - 1, j] - X[i - 1, j + 1] + X[i + 1, j - 1] + 2 * X[i + 1, j] + X[
                i + 1, j + 1]
            dy = -1 * X[i - 1, j - 1] - 2 * X[i, j - 1] - X[i + 1, j - 1] + X[i - 1, j + 1] + 2 * X[i, j + 1] + X[
                i + 1, j + 1]
            newGray[i, j] = int(abs(dx) + abs(dy))  # 用|dx|+|dy| 代替 (dx^2+dy^2)^0.5 减少计算量
    newGray *= (L - 1)
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
    newGray /= (L - 1)
    rows, cols = len(X), len(X[0])
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            dx = -1 * X[i - 1, j - 1] - X[i - 1, j] - X[i - 1, j + 1] \
                 - X[i, j - 1] + 8 * X[i, j] - X[i, j + 1] \
                 - X[i + 1, j - 1] - X[i + 1, j] - X[i + 1, j + 1]
            dy = -1 * X[i - 1, j - 1] - X[i - 1, j] - X[i - 1, j + 1] \
                 - X[i, j - 1] + 8 * X[i, j] - X[i, j + 1] \
                 - X[i + 1, j - 1] - X[i + 1, j] - X[i + 1, j + 1]
            newGray[i, j] = int(abs(dx) + abs(dy))  # 用|dx|+|dy| 代替 (dx^2+dy^2)^0.5 减少计算量
    newGray *= (L - 1)
    return newGray


def color2gray(X):
    """
    彩色图片转成灰度值
    Y = 0.299*R + 0.587*G + 0.114*B
    :param X: 三维的图片
    :return: 转换成的灰度值
    """
    if len(X.shape) == 3:
        tmp = np.array([0.114, 0.587, 0.299])  # 实际图片存储按BGR顺序存储
        return np.floor(np.dot(X, tmp) + 0.5).astype(np.uint8)  # +0.5向下取整，注意float-->uint8
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
    Y = X * (L - 1)
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
        newPGrays[i] = int(math.floor((L - 1) * s + 0.5))

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
        for i in range(1, len(hist) - 1):
            hist[i] = (hist[i - 1] + hist[i] + hist[i + 1]) / 3
        hist[-1] = (hist[-1] + hist[-1] + hist[-2]) / 3

        # 判断是否双峰图像，找出阈值
        threshold = None
        if not _is_dimode(hist):  # 不是双峰
            print('smoothing', t)
            continue
        else:  # 双峰
            if threshold_type == 'mean':
                peak1, peak2 = None, None
                for i in range(1, len(hist)):
                    if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                        if not peak1:
                            peak1 = i
                        else:
                            peak2 = i
                            threshold = (peak1 + peak2) // 2
                            break
            elif threshold_type == 'low':
                peakFound = False
                for i in range(1, len(hist)):
                    if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                        peakFound = True
                    if peakFound and hist[i] <= hist[i - 1] and hist[i] <= hist[i + 1]:
                        threshold = i
                        break
            else:
                raise Exception('No such threshold type:', threshold_type)

            # 二值化
            if threshold:
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Y[i, j] = L - 1 if X[i, j] >= threshold else 0
                # 退出外层1000循环
                break
    return Y


def _is_dimode(hist):
    """判断直方图是否双峰"""
    nPeak = 0
    for i in range(1, len(hist) - 1):
        if hist[i - 1] < hist[i] and hist[i] > hist[i + 1]:
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
    for i in range(gap, X.shape[0] - gap + 1):
        for j in range(gap, X.shape[1] - gap + 1):
            tmp = X[i - gap:i + gap + 1, j - gap:j + gap + 1]
            if way == 'Dilation':
                val = np.max(tmp)
            elif way == 'Frosion':
                val = np.min(tmp)
            else:
                val = X[i, j]  # TODO: 可以补充模板
            Y[i, j] = val
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


def moment(X, p, q):
    """
    二维连续矩 m(p,q) = sumsum(x**p * y**q * gray(x,y))
    :param X: gray
    :return: m(p,q)
    """
    if p < 0 or q < 0:
        raise Exception('p or q must bigger than 0.')
    Y = X.copy().astype(np.float64)
    tmp = 0
    for x in range(len(X)):
        for y in range(len(X[0])):
            Y[x, y] = X[x, y] * 1.0 * (x ** p) * (y ** q)
    return np.sum(Y)


def center_moment(X, p, q):
    """
    中心矩 --> sumsum((x - xx)**p * (y - yy)**q * gray(x, y))
    其中 xx = m(1, 0) / m(0, 0), yy = m(0, 1) / m(0, 0) 代表重心
    m(0, 0) --> 类似于质量
    """
    xx = moment(X, 1, 0) / moment(X, 0, 0)
    yy = moment(X, 0, 1) / moment(X, 0, 0)
    Y = X.copy()
    tmp = 0
    for x in range(len(X)):
        for y in range(len(X[0])):
            Y[x, y] = X[x, y] * 1.0 * ((x - xx) ** p) * ((y - yy) ** q)
    return np.sum(Y)


def humoments(X):
    """
    hu矩 = 不变矩(1 -- 7)
    """
    eta = center_moment  # 简写方法名
    m = moment
    xx = m(X, 1, 0) / m(X, 0, 0)
    yy = m(X, 0, 1) / m(X, 0, 0)
    m00, m01, m02, m03 = m(X, 0, 0), m(X, 0, 1),m(X, 0, 2),m(X, 0, 3)
    m10, m11, m12, m13 = m(X, 1, 0), m(X, 1, 1),m(X, 1, 2),m(X, 1, 3)
    m20, m21, m22, m23 = m(X, 2, 0), m(X, 2, 1),m(X, 2, 2),m(X, 2, 3)
    m30 = m(X, 3, 0)
    # eta
    y00 = m00
    y10 = 0
    y01 = 0
    y11 = m11 - yy * m10
    y20 = m20 - xx * m10
    y02 = m02 - yy * m01
    y21 = m21 - 2 * xx * m11 - yy * m20 + 2 * xx * xx * m01
    y12 = m12 - 2 * yy * m11 - xx * m02 + 2 * yy * yy * m10
    y30 = m30 - 3 * xx * m20 + 2 * xx * xx * m10
    y03 = m03 - 3 * yy * m02 + 2 * yy * yy * m01

    # 归一化 npq = hpq / (h00)**lmd
    # 其中 lmd = (p + q) / 2 + 1
    n20 = y20 / m00 ** 2
    n02 = y02 / m00 ** 2
    n11 = y11 / m00 ** 2
    n30 = y30 / m00 ** 2.5
    n03 = y03 / m00 ** 2.5
    n12 = y12 / m00 ** 2.5
    n21 = y21 / m00 ** 2.5
    # 不变矩(1 -_ 7)
    invariant_moment = [
        n20 + n02,
        (n20 - n02) * (n20 - n02) + 4 * n11 * n11,
        (n30 - 3 * n12) * (n30 - 3 * n12) + (3 * n21 - n03) * (3 * n21 - n03),
        (n30 + n12) * (n30 + n12) + (n21 + n03) * (n21 + n03),
        (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) * (n30 + n12) - 3 * (n21 + n03) * (n21 + n03)) + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03)),
        (n20 - n02) * ((n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03)) + 4 * n11 * (n30 + n12) * (n21 + n03),
        (-1) * ((3 * n21 - n03) * (n30 + n12) * ((n30 + n12) * (n30 + n12) - 3 * (n21 + n03) *(n21 + n03) ) + (3 * n12 - n30) * (n21 + n03) * (3 *(n30 + n12) *(n30 + n12) - (n21 + n03) * (n21 + n03))),
    ]   # 修正不变矩7 --> 乘以(-1)（根据cv2.HuMoments结果)

    res = []
    for im in invariant_moment:
        res.append(im)
    return res

def object_spot(X, X_spot, incr=10):
    """
    对象识别函数，在主图片中识别其中的小部件
    提取特征：7个不变矩
    损失函数：欧氏距离
    :param X: 主图像
    :param X_spot: 需要搜索对象
    :param incr: 边长增量
    :return: 最小损失，坐标点，边长
    """
    # 欧式距离作为损失
    def get_loss(X, Y):
        return np.linalg.norm(X - Y)

    # 一些参数
    max_row, max_col = X.shape  # 主图片长和宽
    start_side = max(X_spot.shape) // 10  # 搜索边长
    n_case = (min(max_row, max_col) - start_side) // incr + 1  # 总共改变边长次数
    losses = np.zeros(n_case)  # 每搜索边长的最小损失
    min_loc = []  # 对应的坐标

    # 小部件
    hu_spot = cv2.HuMoments(cv2.moments(color2gray(X_spot))).ravel()

    # 外循环
    for t in range(n_case):
        # 生成边长和内循环次数
        side = start_side + t * incr
        n_row, n_col = max_row - side + 1, max_col - side + 1

        # 初始化最小损失
        min_loss = get_loss(cv2.HuMoments(cv2.moments(X[0:side, 0:side])).ravel(), hu_spot)
        loc = (0, 0)

        # 寻找最小的损失坐标
        for x in range(n_row):
            for y in range(n_col):
                tmpX = X[x:x+side, y:y+side]
                hu_tmpX = cv2.HuMoments(cv2.moments(tmpX)).ravel()
                tmp_loss = get_loss(hu_tmpX, hu_spot)
                if tmp_loss < min_loss:
                    min_loss = tmp_loss
                    loc = (x, y)

        # 记录当前边长的损失
        losses[t] = min_loss
        min_loc.append(loc)

    # 返回最小损失，坐标，边长
    return np.min(losses), min_loc[np.argmin(losses)], np.argmin(losses) * incr + side


def ssda(X, X_spot, threshold):
    """
    序贯相似性检测（SSDA）
    实现检测子图，暂不能实现大小、方向不相同的子图
    :param X: 要进行匹配的原图片灰度值
    :param X_spot: 要识别的图片灰度值
    :param threshold: 不变阈值
    :return: 匹配的坐标
    """
    # S为原图像，T为要识别图像
    S, T = X.copy(), X_spot.copy()
    I = np.zeros_like(S)    # 最终的SSDA检测曲面矩阵
    if T.shape > S.shape:
        raise Exception('X_spot must smaller than X.')

    if threshold <= 0:
        raise Exception('Threshold must bigger than 0.')

    rows = S.shape[0] - T.shape[0]
    cols = S.shape[1] - T.shape[1]
    size = T.shape
    xs, ys = np.arange(size[0]), np.arange(size[1])
    T_mean = np.mean(T)
    for r in range(rows):
        for c in range(cols):
            # 从原图片中抽出子图
            sub = S[r:r+size[0], c:c+size[1]]
            sub_mean = np.mean(sub)

            # 误差矩阵, 因为阈值很小就可以识别，所以改用循环实现
            # E = np.abs(sub - sub_mean - T + T_mean)
            error = 0   # 累计误差

            # 随机选取点，填充检测曲面I
            np.random.shuffle(xs)
            np.random.shuffle(ys)
            count = 0
            for i in range(len(xs)):
                x, y = xs[i], ys[i]
                error += abs(sub[x, y] - sub_mean - T[x, y] + T_mean)
                count += 1
                if error >= threshold:
                    break
            # 记录(r, c)点的检测曲面值
            I[r, c] = count

    # 返回检测曲面上最大点的值，以及坐标
    max_count = np.max(I)
    return max_count, np.argmax(I) // S.shape[1], np.argmax(I) % S.shape[1]



def reverse(X, L=_L_DEFAULT):
    """取反"""
    Y = X.copy()
    L_matrix = np.ones_like(Y) * (L - 1)
    L_matrix.astype(np.int32)
    return L_matrix - Y
