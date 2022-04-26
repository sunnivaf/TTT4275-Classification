import numpy as np
from sklearn import datasets
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sigmoid import *


features = np.array([0,1,2,3])
#    0. sepal length in cm
#    1. sepal width in cm
#    2. petal length in cm
#    3. petal width in cm

class1 = []
with open("class_1") as f:
    for line in f:
        class1.append([float(line[0:3]), float(line[4:7]), float(line[8:11]), float(line[12:15])])
    class1 = np.array(class1)
#print(class1)

class2 = []
with open("class_2") as f:
    for line in f:
        class2.append([float(line[0:3]), float(line[4:7]), float(line[8:11]), float(line[12:15])])
    class2 = np.array(class2)
#print(class2)

class3 = []
with open("class_3") as f:
    for line in f:
        class3.append([float(line[0:3]), float(line[4:7]), float(line[8:11]), float(line[12:15])])
    class3 = np.array(class3)
#print(class3)

trainStart = 0
trainEnd = 30
trainSize = 30
testStart = 30
testEnd = 50
testSize = 20

C = 3                       # Number of classes
D = 4                       # Number of features

# Tuning variables
iter = 3000
alpha = 0.01

trainSet1 = class1[trainStart:trainEnd, features]
trainSet2 = class2[trainStart:trainEnd, features]
trainSet3 = class3[trainStart:trainEnd, features]
trainSet = np.concatenate((trainSet1,trainSet2,trainSet3),axis=0)
trainSet = np.reshape(trainSet,[trainSize*C,D])

testSet1 = class1[testStart:testEnd, features]
testSet2 = class2[testStart:testEnd, features]
testSet3 = class3[testStart:testEnd, features]
testSet = np.concatenate((testSet1,testSet2,testSet3),axis=0)
testSet = np.reshape(testSet,[(testSize)*C,D])

# Creating targets for each class
target1 = np.tile([1,0,0],trainEnd)
target2 = np.tile([0,1,0],trainEnd)
target3 = np.tile([0,0,1],trainEnd)
target = np.concatenate((target1,target2,target3),axis=None)
target = np.reshape(target,[trainEnd*C,C])

Wx = np.zeros((C,D))        # The weighting matrix
wo = np.zeros((C,1))        # The bias vector
W = np.concatenate((Wx, wo), axis = 1)
N = len(trainSet1)           # Size of the training set -30-
M = len(testSet1)            # Size of the test set -20-

# Confidence vector for each observation
def forwardPred(x, W, N, C):
    g = np.zeros([N*C, C])
    for k, xk in enumerate(x):
        xk = np.append([xk], [1])
        xk = xk.reshape(D+1, 1)
        zk = W@xk
        g[k] = sigmoid(zk.T[0])
    return g

##---------- Training the linear classifier ----------##
def linearClassifier(alpha, iter, N, D, W, C, trainSet, target):
    loss = np.zeros([iter,1],dtype=float)
    for m in range(iter):
        print(f"Training iteration: {m}")
        gradMSE = np.zeros((C,D+1))
        g = forwardPred(trainSet, W, N, C)
        for (xk, gk, tk) in zip(trainSet, g, target): # xk: , tk: class label
            xk = np.append([xk], [1])
            xk = xk.reshape(D+1, 1)
            gradMSE += (((gk-tk)*gk).reshape(C,1) * (np.ones((C,1))-gk.reshape(C,1))) @ xk.T
        W = W - alpha*gradMSE
        loss[m] = mean_squared_error(g,target)
    return W, loss

def confusionMatrix(N, C, trainSet, W):
    print(N)
    confusionMatrix = np.zeros((C,C))
    g = forwardPred(trainSet, W, N, C)
    print(g)
    trueLabel = -1
    for k, gk in enumerate(g):
        if k % N == 0:
            trueLabel += 1
        predLabel = np.argmax(gk)
        confusionMatrix[trueLabel][predLabel] +=1
    return confusionMatrix

def errorRate(confusionMatrix):
    errors = 0
    numSamples = 0
    for trueLabel in range(C):
        for predLabel in range(C):
            if trueLabel != predLabel:
                errors += confusionMatrix[trueLabel][predLabel]
            numSamples += confusionMatrix[trueLabel][predLabel]
    return errors/numSamples

# def removeFeature(trainSet, testSet, alpha, iter, N, D, W, C, target, feature):
#     W = np.zeros([C, D+1],dtype=float)
#     trainSet = np.delete(trainSet, 1, 1)
#     testSet = np.delete(testSet, 1, 1)
#     W,_ = linearClassifier(alpha, iter, N, D, W, C, trainSet, target)

def plotAlphas():
    alphas = [0.05,0.01,0.005,0.001]
    for a in alphas:
        W = np.zeros([C, D+1],dtype=float)
        W,loss = linearClassifier(a, iter, N, D, W, C, trainSet, target)
        plt.plot(np.arange(iter),loss,label="alpha= "+str(a))
        plt.legend()
        plt.grid()
        plt.xlabel("number of iterations")
        plt.title("MSE")
    plt.show()

def plotConfusionMatrix(title, confusionMatrix):
    group_counts = ["{0:0.0f}".format(value) for value in
                    confusionMatrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        confusionMatrix.flatten()/np.sum(confusionMatrix)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
            zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(3,3)
    ax = sn.heatmap(confusionMatrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label');
    ax.xaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])
    ax.yaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])
    plt.show()

def featureName(f):
    if f == 0: return "Sepal length"
    if f == 1: return "Sepal width"
    if f == 2: return "Petal length"
    if f == 3: return "Petal width"

def plotHistograms():
    for f in features:
        plt.figure(f)
        plt.hist(class1[:,f], alpha=0.5, label = "Setosa")
        plt.hist(class2[:,f], alpha=0.5, label = "Versicolour")
        plt.hist(class3[:,f], alpha=0.5, label = "Virginica")
        plt.legend()
        plt.xlabel('cm')
        plt.ylabel('count')
        plt.grid()
        plt.title(featureName(f))
    plt.show()

if __name__ == '__main__':
    W,_ = linearClassifier(alpha, iter, N, D, W, C, trainSet, target)
    #print(len(trainSet1))
    #print(trainSet)
    #forwardPred(trainSet, W, N, C)
    # c_train = confusionMatrix(N, C, trainSet, W)
    # plotConfusionMatrix('Training set', c_train)
    # c_test = confusionMatrix(M, C, testSet, W)
    # plotConfusionMatrix('Test set', c_test)
    #plotHistograms()
    plotAlphas()

        