#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *
from numpy import linalg as la

def svdTest():
    U, sigma, VT = la.svd([[1, 1], [7, 7]])
    print(U)
    print(sigma)
    print(VT)
    pass

# def loadExData():
#     return [[1, 1, 1, 0, 0],
#             [2, 2, 2, 0, 0],
#             [1, 1, 1, 0, 0],
#             [5, 5, 5, 0, 0],
#             [1, 1, 0, 2, 2],
#             [0, 0, 0, 3, 3],
#             [0, 0, 0, 1, 1]]

def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

# inA,inB 列向量
# 欧式距离
def euclidSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))

# 皮尔逊相关系数
def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]

# 余弦相似度
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num/denom)

# 计算在给定相似度计算方法的条件下，用户对物品的估计评分值
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating ==0:
            continue
        #寻找两个用户都评级的物品
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print("the %d and %d similarity is: %f" % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

# 推荐引擎
# standEst包括 数据矩阵、用户编号、物品编号和相似度计算方法
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 寻找未评级物品
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    # 寻找前N个未评级物品
    return sorted(itemScores, key=lambda jj:jj[1], reverse=True)[:N]

def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma, VT = la.svd(dataMat)
    # 建立对角矩阵
    Sig4 = mat(eye(4) * Sigma[:4])
    # 构建转换后的物品
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        print("the %d and %d similarity is: %f" % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

# 打印矩阵，矩阵包含浮点数，必须定义浅色和深色
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1,)
            else:
                print(0,)
        print('')

# 图像的压缩
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("*** original matrix ***")
    printMat(myMat, thresh)

    U, Sigma, VT = la.svd(myMat)
    # 将sigma矩阵化
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("*** reconstructed matrix using %d singular valuse ***" % numSV)
    printMat(reconMat, thresh)

def decomposeTest():
    Data = loadExData()
    U, Sigma, VT = la.svd(Data)
    # print(U)
    print(Sigma)
    # print(VT)

    Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    uu = U[:, :3] * Sig3 * VT[:3, :]

    print(uu)

def simTest():
    myMat = mat(loadExData())
    simeuc = euclidSim(myMat[:, 0], myMat[:, 4])
    print("欧式距离：",simeuc)
    simecuc2 = euclidSim(myMat[:, 0], myMat[:, 0])
    print("欧式距离：",simecuc2)

    simcos = cosSim(myMat[:, 0], myMat[:,4])
    print("余弦相似度：", simcos)
    simcos2 = cosSim(myMat[:, 0], myMat[:, 0])
    print("余弦相似度：", simcos2)

    simpears = pearsSim(myMat[:, 0], myMat[:, 4])
    print("皮尔逊相似度：", simpears)
    simpears2 = pearsSim(myMat[:, 0], myMat[:, 0])
    print("皮尔逊相似度：", simpears2)

def recommendTest():
    myMat = mat(loadExData())
    myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
    myMat[3, 3] = 2
    print(myMat)

    a = recommend(myMat, 2)
    print(a)
    print("--------------------------------------")
    b = recommend(myMat, 1, estMethod=svdEst)
    print(b)
    print("--------------------------------------")
    c = recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim)
    print(c)

if __name__ == '__main__':
    # svdTest()
    # decomposeTest()
    # simTest()
    # recommendTest()
    imgCompress(2)