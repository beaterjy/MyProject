import numpy as np


class DataSet():
    """
    保存数据集的数据和标签
    """

    def __init__(self):
        super().__init__()
        self._datas = []
        self._tags = []

    def add_sample(self, data, tag):
        """
        添加新样本
        """
        self._datas.append(data)
        self._tags.append(tag)

    def get_datas(self):
        """
        获取数据集数据矩阵
        :return: np.ndarray
        """
        return np.array(self._datas)

    def get_tags(self):
        """
        获取标签数据矩阵
        :return: np.ndarray
        """
        return np.array(self._tags)

    def __len__(self):
        """返回当前数据集的样本数量"""
        return len(self._datas)
