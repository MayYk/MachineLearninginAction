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


def generateRules(L, supportData, minConf=0.7):
    """

    :param L:频繁项集列表
    :param supportData:包含频繁项集列表支持数据的字典
    :param minConf:最小可信度阈值
    :return:
    """
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConSeq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

# 对规则进行评估
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:',conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

# 生成候选规则集合
def rulesFromConSeq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > m+1:
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if len(Hmp1) > 1:
            rulesFromConSeq(freqSet, Hmp1, supportData, brl, minConf)

# votesmart apikey正在等待审批
# def votesmartData():
#     votesmart.apikey = '123456789'

def mushroomTest():
    mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]
    L,supportData = apriori(mushDataSet, minSupport=0.3)

    for item in L[1]:
        if item.intersection('2'):
            print(item)

    for item in L[3]:
        if item.intersection('2'):
            print(item)


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
    L, supportData = apriori(dataSet)
    print(L[0])
    print(L[1])
    print(L[2])
    print(L[3])

def rulesTest():
    dataSet = loadDataSet()
    L, supportData = apriori(dataSet, minSupport=0.5)

    rules = generateRules(L, supportData, minConf=0.7)
    print(rules)

    rules = generateRules(L, supportData, minConf=0.5)
    print(rules)

if __name__ == '__main__':
    # supTest()
    # aprioriTest()
    # rulesTest()
    mushroomTest()