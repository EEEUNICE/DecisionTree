#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
from math import log
import operator
import matplotlib.pyplot as plt

'''
dataset: 数据集，每一行为一组数据
value：对应的类的值
feature：属性
'''


#计算信息熵
def Ent(dataSet):
    total_number = len(dataSet)
    labelCounts = {}
    for column in dataSet:
        currentLabel = column[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Ent = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / total_number
        Ent -= prob * log(prob, 2)

    return Ent

#对离散值的划分
def splitNotContinuousDataSet(dataSet, axis, value):
    retDataset = []
    for column in dataSet:
        if column[axis] == value:
            reducedColumn = column[:axis]
            reducedColumn.extend(column[axis+1:])
            retDataset.append(reducedColumn)

    return retDataset

#对连续值的划分
def splitContinuousDataSet(dataSet, axis, value, direction):
    retDataset = []
    for column in dataSet:
        if direction == 0:
            if column[axis] > value:
                # reducedColumn = column[:axis]
                # reducedColumn.extend(column[axis+1:])
                retDataset.append(column)
        else:
            if column[axis] <= value:
                # reducedColumn = column[:axis]
                # reducedColumn.extend(column[axis+1:])
                retDataset.append(column)

    return retDataset

#选择最优的属性划分
def chooseBestFeaturetoSplit(dataSet, features):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = Ent(dataSet)                  #计算信息熵 
    bestGain = 0.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        if type(featureList[0]).__name__ == 'float' or type(featureList[0]).__name__ == 'int':
            sortList = sorted(featureList)
            splitList = []
            for j in range(len(sortList) - 1):
                splitList.append((sortList[j] + sortList[j+1]) / 2.0)
            bestSplitEntropy = 100
            slen = len(splitList)
            for j in range(slen):
                value = splitList[j]
                newEntropy = 0.0
                subDataSet0 = splitContinuousDataSet(dataSet, i ,value, 0)
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(subDataSet0) / float(len(dataSet))
                newEntropy += prob0 * Ent(subDataSet0)
                prob1 = len(subDataSet1) / float(len(dataSet))
                newEntropy += prob1 * Ent(subDataSet1)
                if newEntropy < bestSplitEntropy:
                    bestSplitEntropy = newEntropy
                    bestSplit = j
            bestSplitDict[features[i]] = splitList[bestSplit]
            Gain = baseEntropy - bestSplitEntropy
        else:
            uniqueVals = set(featureList)
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = splitNotContinuousDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob*Ent(subDataSet)
            Gain = baseEntropy - newEntropy
        if Gain > bestGain:
            bestGain = Gain
            bestFeature = i
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[features[bestFeature]]
        features[bestFeature] = features[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][bestFeature] < bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0

    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


'''
START 生成决策树  
'''

def creatTree(dataSet, features, data_full, features_full):
    classList = [example[-1] for example in dataSet]
    
    #data中样本全属于同一类别 递归返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
   #当所有的特征都用完时，采用多数表决的方法来决定该叶子节点的分类
   #即该叶节点中属于某一类最多的样本数，那么我们就说该叶节点属于那一类！   递归返回
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
 
    bestFeat = chooseBestFeaturetoSplit(dataSet, features)
    bestFeatFeature = features[bestFeat]
    mytree = {bestFeatFeature:{}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        currentLabel = features_full.index(features[bestFeat])
        featValuesFull = [example[currentLabel] for example in data_full]
        uniqueValsFull = set(featValuesFull)
    del(features[bestFeat])
    for value in uniqueVals:
        subFeature = features[:]
        if type(dataSet[0][bestFeat]).__name__ == 'str':
            uniqueValsFull.remove(value)
        mytree[bestFeatFeature][value] = creatTree(splitNotContinuousDataSet(dataSet, bestFeat, value), subFeature, data_full, features_full)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        for value in uniqueValsFull:
            mytree[bestFeatFeature][value] = majorityCnt(classList)

    return mytree
'''
END 生成决策树  
'''



'''
START 决策树字典图像化  
'''

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
			
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()
'''
END 决策树字典图像化  
'''


df = pd.read_csv('watermelon_4_2.csv')#读入数据
data = df.values[:, 1:].tolist()#将数值换成数组
data_full = data[:]#数值copy

features = df.columns.values[1:-1].tolist()#将特征值标签存入数组
features_full = features[:]#特征值数组copy


myTree = creatTree(data, features, data_full, features_full)
#print( myTree)
createPlot(myTree)
