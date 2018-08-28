# 利用sklearn学习库：对二元数据进行逻辑回归预测

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np


def logisticRegression():
    data = loadtxtAndcsv_data("data1.txt", ",", np.float64)
    X = data[:, 0:-1]
    y = data[:, -1]

    # 将数据源分为训练集合和测试集合
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 将数据进行标准化
    scaler = StandardScaler()
    scaler.fit(x_train)
    # 利用训练集合的均值和标准差把训练集合和测试集合进行数据标准化
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    # 利用sklearn 的逻辑回归模型
    model = LogisticRegression()
    # 进行逻辑回归训练
    model.fit(x_train, y_train)

    # 对测试集合进行预测
    predict = model.predict(x_test)
    # 统计测试结果的正确率
    right = sum(predict == y_test)

    # 将测试对比结果输出
    predict = np.hstack((predict.reshape(-1, 1), y_test.reshape(-1, 1)))
    print(predict)
   # print predict
    rightRate = right*100.0/predict.shape[0]
    print(rightRate)


def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


def loadnpy_data(fileName):
    return np.load(fileName)


if __name__ == "__main__":
    logisticRegression()
