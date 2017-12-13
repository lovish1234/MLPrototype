import numpy as np
from numpy import load as np_load

# classifiers
import sklearn
from sklearn import metrics
from sklearn.svm import LinearSVC

# plot using matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import PIL.Image
from cStringIO import StringIO

import random
import os
import struct
import itertools
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--showImage', required=False, default=0, type=int, help='show figures')

params = parser.parse_args()

def readIdx(filename):
    ''' 
    A function to read in a numpy array from MNIST/ FashionMNIST / notMNIST files 
    Borrowed from Tyler Neylon
    
    https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    '''
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        images = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
    return images 

def loadDataset(dataset, classes, numberOfInstances, sizeOfImage):
    '''
    Load a particular dataset for testing GAN
    '''
    if dataset=='MNIST':
        folder = '../data/MNIST/raw/'
    elif dataset=='FashionMNIST':
        folder = '../data/FashionMNIST/raw/'
    elif dataset=='notMNIST':
        folder = '../data/notMNIST/raw/'
    
    data = {}
    fileName = os.listdir(folder)
    
    for i in range(len(fileName)):
        fileName[i]=folder+fileName[i]
        if 'train-images' in fileName[i]:
            data['images'] = readIdx(fileName[i])
        elif 'train-labels':

            data['labels'] = readIdx(fileName[i])
            
    images = data['images']
    labels = data['labels']
    
    # random shuffling of images and labels
    p = np.random.permutation(images.shape[0])
    images = images[p]
    labels = labels[p]
    
    # get images only belonging to a particular class
    imagesByClass = images[np.where(labels==classes)]
    labelsByClass = labels[np.where(labels==classes)]
    
    # number of images to be extracted
    imagesByClass = imagesByClass[0:numberOfInstances]
    labelsByClass = labelsByClass[0:numberOfInstances]

    imagesByClassResize = np.zeros((imagesByClass.shape[0],
                                    sizeOfImage,
                                    sizeOfImage))
    
    for i in range(imagesByClass.shape[0]):
        tempImage = PIL.Image.fromarray(np.uint8(imagesByClass[i,:,:]))
        tempImage = tempImage.resize((sizeOfImage,sizeOfImage),
                                     PIL.Image.ANTIALIAS)
        imagesByClassResize[i,:,:] = np.asarray(tempImage)
    
    imagesByClassResize=imagesByClassResize/255.0
    return imagesByClassResize,labelsByClass


def getFakeData(dataSet, fakeClasses, instances):
    fakeClasses =  sorted(fakeClasses)
    imageArray = np.empty((1,4096))
    labelArray = np.empty((1))
    for i in fakeClasses:
        tempImageArray = np_load('../DCGAN/results/'+'compressed'+'/'+dataSet+'/'+dataSet+'_'+str(i)+'_'+str(instances)+'.npy')
        tempImageArray = tempImageArray.reshape(tempImageArray.shape[0],-1)
        tempLabelArray = np.empty(tempImageArray.shape[0])
        tempLabelArray.fill(i)        
        imageArray = np.concatenate([imageArray, tempImageArray])
        labelArray = np.concatenate([labelArray, tempLabelArray])
    imageArray = imageArray[1:]
    labelArray = labelArray[1:]
    
    # random shuffling of images and labels
    p = np.random.permutation(imageArray.shape[0])
    imageArray = imageArray[p]
    labelArray = labelArray[p]
    
    return imageArray, labelArray

def getRealData(dataSet, realClasses, instances):
    realClasses = sorted(realClasses)
    imageArray = np.empty((1,4096))
    labelArray = np.empty((1))
    for i in realClasses:
        tempImageArray,tempLabelArray = loadDataset(dataSet, i, instances, 64)
        tempImageArray = tempImageArray.reshape(tempImageArray.shape[0],-1)
        imageArray = np.concatenate([imageArray, tempImageArray])
        labelArray = np.concatenate([labelArray, tempLabelArray])
    imageArray = imageArray[1:]
    labelArray = labelArray[1:]
    
    # random shuffling of images and labels
    p = np.random.permutation(imageArray.shape[0])
    imageArray = imageArray[p]
    labelArray = labelArray[p]
    
    return imageArray, labelArray

def plotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Reds'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        w = ' '+str(format(cm[i, j], fmt))+' '
        plt.text(j, i, w,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plotAccuracy(dataSet, trainSet, accuracyList, showImage):
    
    xaxis = [10,100,1000]
    yaxis = accuracyList
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(xaxis,yaxis,'r--',label=dataSet)
    plt.plot(xaxis,yaxis,'rs')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    plt.ylabel('Accuracy(%)')
    plt.xlabel('Number of Instances(log)')
    plt.xscale('log')
    plt.tight_layout()
    plt.title('')
    ax.legend()
    
    plotFileName = 'plots'+'/'+'accuracy'+'/'+dataSet+'/'+dataSet+'_'+trainSet+'_real.png'
    plt.savefig(plotFileName, bbox_inches='tight')
    if showImage==1:
        plt.show()
    plt.close()

def getClasses(dataSet):
    if dataSet=='MNIST':
        class_names = [0,1,2,3,4,5,6,7,8,9]
    elif dataSet=='FashionMNIST':
        class_names = ['T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    elif dataSet=='notMNIST':
        class_names = ['A','B','C','D','E','F','G','H','I','J']
    return class_names

# train a multi-class SVM
def classify(dataSet, trainSet):
    
    class_names = getClasses(dataSet)
    noOfInstances = [10,100,1000]
    accuracyList = []
    np.set_printoptions(precision=2)
    print ('============================')
    print ('Dataset: ' + str(dataSet))
    for i in noOfInstances:
        if trainSet == 'fake':
            trainData, trainLabels = getFakeData(dataSet, [0,1,2,3,4,5,6,7,8,9], i)
        elif trainSet == 'real':
            trainData, trainLabels = getRealData(dataSet, [0,1,2,3,4,5,6,7,8,9], i)
            
        testData, testLabels = getRealData(dataSet, [0,1,2,3,4,5,6,7,8,9], 1000)
        
        # multi-class SVM
        clf = LinearSVC()
            
        clf.fit(trainData, trainLabels)
        predictions = clf.predict(testData)
        accuracy = sklearn.metrics.accuracy_score(testLabels, predictions)
        accuracyList.append(accuracy)
        
        print ('=================================')
        print ('Number of Instances: ' + str(i))
        print ('Accuracy: ' + str(accuracy))

        confusionMatrix = sklearn.metrics.confusion_matrix(testLabels, predictions, [0,1,2,3,4,5,6,7,8,9])

        # Plot normalized confusion matrix
        fig = plt.figure()
        plotConfusionMatrix(confusionMatrix,
                              classes=class_names,
                              normalize=True,
                              title='Normalized Confusion matrix')

        
        fileName = dataSet+'_'+trainSet+'_'+str(i)
        plotFileName = 'plots'+'/'+'cm'+'/'+dataSet+'/'+fileName+'.png'
        plt.savefig(plotFileName, bbox_inches='tight')
        if params.showImage==1:
            plt.show()
        plt.close()
    
    plotAccuracy(dataSet, trainSet, accuracyList, params.showImage)
    
if __name__=='__main__':
    classify('MNIST','fake')
    classify('FashionMNIST','fake')
    classify('notMNIST','fake')
    
    classify('MNIST','real')
    classify('FashionMNIST','real')
    classify('notMNIST','real')