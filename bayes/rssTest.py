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
# 教程链接不可用，修改措施
# http://brittanyherself.com/cgg/tutorial-how-to-subscribe-to-craigslists-rss-feeds/

# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf = feedparser.parse('http://sybay.craigslist.org/stp/index.rss')
ny = feedparser.parse('https://newyork.craigslist.org/search/bts?format=rss')
sf = feedparser.parse('https://syracuse.craigslist.org/search/acc?format=rss')

vocabList, pSF, pNY = localWords(ny, sf)

