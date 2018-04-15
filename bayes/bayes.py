#!user/bin/env python
# _*_ coding:utf-8 _*_
from numpy import *

# 词表到向量的转换函数
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1代表侮辱性文字，0代表正常言论
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # |按位或,创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 构建词向量的方法，只记录了每个词是否出现，而没有记录词出现的次数，这样的模型叫做词集模型
# 如果在词向量中记录词出现的次数，没出现一次，则多记录一次，这样的词向量构建方法，被称为词袋模型

# 词集模型
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个其中所含元素都为0的向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec

# 朴素贝叶斯分类器训练函数
# 难理解：
# 讲解：https://blog.csdn.net/moxigandashu/article/details/71480251?locationNum=16&fps=1
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # p0Num = zeros(numWords);    p1Num = zeros(numWords)
    # p0Denom = 1.0;    p1Denom = 1.0
    p0Num = ones(numWords);    p1Num = ones(numWords)
    p0Denom = 2.0;    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 统计所有类别为1的词条向量中各个词条出现的次数
            p1Num += trainMatrix[i]
            # 统计类别为1的词条向量中出现的所有词条的总数
            # 即统计类1所有文档中出现单词的数目
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num/p1Denom
    # p0Vect = p0Num/p0Denom
    # 数值过小，下溢出，取自然对数
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

# 朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 文件解析
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower for tok in listOfTokens if len(tok) > 2]

# 完整的垃圾邮件测试函数
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        # 导入并解析文本
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        # email\ham中的23.txt中第二段多了一个问号，导致解码失败，删除‘？’之后便可以继续执行。
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    # 留存交叉验证(hold-out cross validation)
    # 10封邮件被随机选择添加到测试集，从训练集中删除
    for i in range(10):
        # random.uniform(x, y)方法将随机生成下一个实数，它在 [x, y) 范围内。
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 计算分类所需概率
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount)/len(testSet))

testingNB()
# spamTest()