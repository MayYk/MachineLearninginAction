#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
# 矩阵求导：https://zhuanlan.zhihu.com/p/24709748

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

# 计算最佳拟合直线
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

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 创建对角矩阵
    weights = mat(eye((m)))
    for j in range(m):
        # 权重值大小以指数及衰减
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def plotStandRegres(xArr,yArr):
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

def plotLwlr(xArr, yArr):
    yHat_0 = lwlrTest(xArr, xArr, yArr, 1.0)
    yHat_1 = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.003)

    xMat = mat(xArr)
    yMat = mat(yArr)
    # 排序，返回索引值
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    # 中文显示的bug
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10, 8))
    axs[0].plot(xSort[:, 1], yHat_0[srtInd], c='red')  # 绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_1[srtInd], c='red')  # 绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_2[srtInd], c='red')  # 绘制回归曲线
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title('k=1.0')
    axs1_title_text = axs[1].set_title('k=0.01')
    axs2_title_text = axs[2].set_title('k=0.003')
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()

if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    # print(xArr[0:2])
    # plotStandRegres(xArr,yArr)

    # lwlr(xArr[0], xArr, yArr, 1.0)
    # lwlr(xArr[0], xArr, yArr, 0.001)
    plotLwlr(xArr,yArr)
