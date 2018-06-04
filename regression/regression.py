#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # np.linalg.det()：矩阵求行列式（标量）
    # np.linalg.inv()：矩阵求逆
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do i inverse")
        return
    # .I求逆矩阵
    # ws = linalg.solvs, xMat.T * yMat)
    ws = xTx.I * (xMat.T * yMat)
    return ws

if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    # print(xArr[0:2])
    ws = standRegres(xArr, yArr)
    print(ws)

    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # flatten：用于array和mat的降维
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()


