import csv
import random
import math
import operator
import pandas as pd

'''def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'rb')as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]=float(dataset[x][y])
            if random.random()<split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
                '''
#计算距离，也就是所预测的点和训练集中的点的距离
def eulideanDistance(instance1,instance2,length):
    distance=0
    for x in range(length):
        distance+=pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)
#计算所要预测的点距离最小的K个邻居
def getNeighbors(trainingSet,testInstance,k):
    distances=[]
    length=len(testInstance)-1
    for x in range(len(trainingSet)):
        dist=eulideanDistance(testInstance,trainingSet[x],length)
        distances.append([trainingSet[x],dist])
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
#把所计算的邻居进行分类，返回最多的那个类，也就是预测点所属于的类
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
       
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1

        sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
        #operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号（即需要获取的数据在对象中的序号）
    return sortedVotes[0][0]
#计算精确度，把预测集中的真实分类和预测的分类对比，计算精确度
def getAccuracy(testSet,predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1]==predictions[x]:
             correct+=1
    return (correct/float(len(testSet))*100.0)

def main():
    trainingSet=[]
    testSet=[]
    with open('iris_training.csv','r')as csvfile:
        lines=csv.reader(csvfile)
        for row in lines:
            dataset=list(row)
        
            #dataset[-1]=int(dataset[-1])
            for y in range(4):
                dataset[y]=float(dataset[y])
                
                trainingSet.append(dataset)
    #print(trainingSet)
    with open('iris_test.csv','r')as csvfile:
        lines=csv.reader(csvfile)
        for row in lines:
            dataset=list(row)
            #dataset[-1]=int(dataset[-1])
            for y in range(4):
                dataset[y]=float(dataset[y])
                testSet.append(dataset)
    
    print('train set:'+str(len(trainingSet)))
    print('test set:'+str(len(testSet)))#repr()和作用一样str()

    predictions=[]
    k=3

    for x in range(len(testSet)):
        neighbors=getNeighbors(trainingSet,testSet[x],k)
        result=getResponse(neighbors)
        predictions.append(result)
        print('predictions'+repr(result)+' ,actual'+repr(testSet[x][-1]))

    accuracy=getAccuracy(testSet,predictions)
    print('accuracy'+repr(accuracy)+'%')

    

main()











        
        
