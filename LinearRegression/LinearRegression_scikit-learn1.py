# -*- coding: utf-8 -*-
# 一元回归预测
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler  # 引入归一化的包


def linearRegression():
    # print("加载数据...\n")
    # 读取数据
    #data = loadtxtAndcsv_data("./data.txt", ",", np.float64)
    # X对应0列和第1列
    # x = np.array(data[:, 0:2], dtype=np.float64)
    # y对应最后一列
    # z = np.array(data[:, 2], dtype=np.float64)
    x = np.array([[1], [2], [3], [4]])
    z = np.array([3, 4.01, 5.2, 5.99])

    # 归一化算子
    scaler = StandardScaler()
    # 计算均值和方差
    scaler.fit(x)
    # 利用均值和标准差对数据进行标准化
    # (数据-均值)/标准差
    x_train = scaler.transform(x)
    # 利用上面的均值和方差对检验数据标准化
    x_test = scaler.transform(np.array([[2.5]]))

    # 二元数据线性模型拟合 x_train 是二元的标准化数据
    model = linear_model.LinearRegression()
    # 拟合器进行拟合，得到数据
    model.fit(x_train, z)

    # 预测结果
    result = model.predict(x_test)
    print(result)
   # print model.coef_       # Coefficient of the features 决策函数中的特征系数
   # print model.intercept_  # 又名bias偏置,若设置为False，则为0
   # print result            # 预测结果

# 加载txt和csv文件


def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)

# 加载npy文件


def loadnpy_data(fileName):
    return np.load(fileName)


if __name__ == "__main__":
    linearRegression()
