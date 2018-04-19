#!user/bin/env python
# _*_ coding:utf-8 _*_
import bayes
from numpy import *
import feedparser
# RSS源分类器及高频词去除函数
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1), reverse = True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    docList = [];
    classList = [];
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        # 每次访问一条RSS源
        wordList = bayes.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = bayes.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)

    # 去掉出现频数最高的词
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam = bayes.trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if bayes.classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print('the error rate is:', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V

# 最具表征性的词汇显示函数
def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -4.5:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -4.5:
            topNY.append((vocabList[i], p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF')
    for item in sortedSF:
        print(item[0])

    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY')
    for item in sortedNY:
        print(item[0])
# 教程链接不可用，修改措施
# 修改RSS城市来源：RSS说明：http://brittanyherself.com/cgg/tutorial-how-to-subscribe-to-craigslists-rss-feeds/
# 或者
# 更换数据网站来源：http://www.cnblogs.com/femaleprogramer/p/3854970.html

# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf = feedparser.parse('http://sybay.craigslist.org/stp/index.rss')
if __name__ == '__main__':
    ny = feedparser.parse('https://newyork.craigslist.org/search/ats?format=rss')
    sf = feedparser.parse('https://syracuse.craigslist.org/search/ats?format=rss')

# vocabList, pSF, pNY = localWords(ny, sf)
    getTopWords(ny, sf)
