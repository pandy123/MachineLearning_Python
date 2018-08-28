# -*- coding: utf-8 -*-
# 逻辑回归预测手写数字
from scipy import io as spio
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression


def logisticRegression_oneVsAll():
    data = loadmat_data("data_digits.mat")
    X = data['X']   # 获取X数据，每一行对应一个数字20x20px
    y = data['y']   # 这里读取mat文件y的shape=(5000, 1)
    y = np.ravel(y)  # 调用sklearn需要转化成一维的(5000,)

   # 创建手回归模型计算算子
    model = LogisticRegression()
   # 进行拟合
    model.fit(X, y)
   # 进行预测
    predict = model.predict(X)
    print(predict)
    rightRatio = np.mean(np.float64(predict == y)*100)
    print(rightRatio)

    # print u"预测准确度为：%f%%"%np.mean(np.float64(predict == y)*100)

# 加载mat文件


def loadmat_data(fileName):
    return spio.loadmat(fileName)


if __name__ == "__main__":
    logisticRegression_oneVsAll()
