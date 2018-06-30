#!user/bin/env python
# _*_ coding:utf-8 _*_

def loadDataSet():
    return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # 对C1中每个项构建一个不变集合
    return list(map(frozenset, C1))

def scanD(D, Ck, minSupport):
    """

    :param D:   数据集
    :param Ck:  候选集
    :param minSupport:  最小支持度
    :return: 频繁项集列表
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.__contains__(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        # 计算所有项集的支持度
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # 前k-2个项相同时，将两个集合合并
            ki = Lk[i]
            L1 = list(Lk[i])[:k-2]
            kj = Lk[j]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        s = L[k-2]
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData



def supTest():
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    print(C1)
    D = list(map(set, dataSet))
    print(D)
    L1, suppData0 = scanD(D, C1, 0.5)
    print(L1)

def aprioriTest():
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet)
    print(L[0])
    print(L[1])
    print(L[2])
    print(L[3])

if __name__ == '__main__':
    # supTest()
    aprioriTest()

