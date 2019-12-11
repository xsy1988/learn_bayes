# Create by MrZhang on 2019-11-20

import numpy as np
import re
from math import log

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(splitSent(document[0]))
    return list(vocabSet)

def splitSent(inputSent):
    return re.split(r'\W+', inputSent)

def createDataSet():

    set1 = ['This is a release of several new models which were the result of an improvement the pre-processing code.', 1]
    set2 = ['We have made two new BERT models available:', 1]
    set3 = ['The data and training were otherwise identical', 1]
    set4 = ['Words are the core meaning bearing units in language.', 1]
    set5 = ['Language modelling is important task of great practical use in many NLP applications.', 1]
    set6 = ['By conditioning an RNN language model on an input representation we can generate contextually relevant language.', 1]
    set7 = ['then cover more recent neural approaches such as DeepMind WaveNet model', 1]
    set8 = ['We will be using Piazza to facilitate class discussion during the course.', 1]
    set9 = ['Here are some helpful instructions to use the latest code', 1]
    set10 = ['Sometimes we will provide updated jars here which have the latest version of the code.', 1]
    set11 = ['None of them noticed a large, tawny owl flutter past the window.', 0]
    set12 = ['It was on the corner of the street that he noticed the first sign of something peculiar', 0]
    set13 = ['He eyed them angrily as he passed.', 0]
    set14 = ['He looked back at the whisperers as if he wanted to say something to them', 0]
    set15 = ['It was now sitting on his garden wall.', 0]
    set16 = ['She told him over dinner all about Mrs.', 0]
    set17 = ['Nothing like this man had ever been seen on Privet Drive.', 0]
    set18 = ['The nearest street lamp went out with a little pop.', 0]
    set19 = ['But how is the boy getting here', 0]
    set20 = ['One small hand closed on the letter beside him and he slept on, not knowing he was special', 0]
    myDataSet = [set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11, set12, set13, set14, set15, set16, set17, set18, set19, set20]
    return myDataSet

def word2Vec(vocabList, inputSet):
    returnVec = np.zeros(len(vocabList))
    # print(returnVec)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: {} is not in my vocabulary".format(word))
    return returnVec

# inputSet = 'This is my new book'
# # inputSet = re.split(r'\W+', inputSet)
# vocabList = createVocabList(createDataSet())
# print(word2Vec(vocabList, splitSent(inputSet)))

def docToMatrix(docData, vocabList):
    numOfData = len(docData)
    sentMatrix = []
    classVec = []

    for i in range(numOfData):
        sentence = docData[i][0]
        sentClass = docData[i][1]
        sentVec = word2Vec(vocabList, splitSent(sentence))
        sentMatrix.append(sentVec)
        classVec.append(sentClass)

    return sentMatrix, classVec

def trainBayes(trainMatrix, trainClass):
    numOfDatas = len(trainClass)
    numOfWords = len(trainMatrix[0])

    probTrueNum = np.ones(numOfWords)
    probFalseNum = np.ones(numOfWords)
    # print(probTrueNum)

    probOfTrueDoc = sum(trainClass) / float(numOfDatas)

    trueDocWordNum = 2.0
    falseDocWordNum = 2.0

    for i in range(numOfDatas):
        if trainClass[i] == 1:
            probTrueNum += trainMatrix[i]
            trueDocWordNum += sum(trainMatrix[i])
        else:
            probFalseNum += trainMatrix[i]
            falseDocWordNum += sum(trainMatrix[i])
    # print(probTrueNum / trueDocWordNum)
    probTrueVec = np.log(probTrueNum / trueDocWordNum)
    probFalseVec = np.log(probFalseNum / falseDocWordNum)

    return probTrueVec, probFalseVec, probOfTrueDoc

def classify(inputVec, probTrueVec, probFalseVec, probOfTrueDoc):
    probOfTrue = sum(inputVec * probTrueVec) + log(probOfTrueDoc)
    probOfFalse = sum(inputVec * probFalseVec) + log(1.0 - probOfTrueDoc)
    if probOfTrue > probOfFalse:
        print(probOfTrue)
        return 1
    else:
        print(probOfFalse)
        return 0

def testBayes(testSent, probTrueVec, probFalseVec, probOfTrueDoc, vocabList):
    testSentVec = word2Vec(vocabList, splitSent(testSent))
    result = classify(testSentVec, probTrueVec, probFalseVec, probOfTrueDoc)
    if result == 1:
        print("This sentence is classified as abs.")
    else:
        print("This sentence is not classified as abs.")

if __name__ == '__main__':
    dataSet = createDataSet()
    vocabList = createVocabList(dataSet)
    sentMatrix, classVec = docToMatrix(dataSet, vocabList)
    probTrueVec, probFalseVec, probOfTrueDoc = trainBayes(sentMatrix, classVec)
    testSent = 'This is he wall all Bert data.'
    testBayes(testSent, probTrueVec, probFalseVec, probOfTrueDoc, vocabList)













