#!user/bin/env python
# _*_ coding:utf-8 _*_
import bayes

listPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.creatVocabList(listPosts)
postZero = bayes.setOfWords2Vec(myVocabList, listPosts[0])
postThree = bayes.setOfWords2Vec(myVocabList, listPosts[3])
print('-------------------------------------------------------')
