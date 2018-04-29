import numpy as np
from numpy import load as np_load

# classifiers
import sklearn
from sklearn import metrics

# plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
%matplotlib inline

from os import fsync
import PIL
from numpy import save as np_save
from numpy import load as np_load

# import datasets
import sys
sys.path.append('../')
from Datasets.loadDataset import loadDataset, getClasses, getHelperClass, checkAndCreateFolder, \
getChannels, getImageSamples, getClassifier, getImageSize
reload(sys.modules[loadDataset.__module__])

# import configuration
sys.path.append('../DCGAN/config.py')
import config
reload(config)
from config import *

from collections import OrderedDict

import random
import itertools

import time


def sync(fh):
    """
    This makes sure data is written to disk, so that buffering doesn't influence the timings.
    """
    fh.flush()
    fsync(fh.fileno())
    
def getInitialArray(dataSet):
    if dataSet=='SVHN' or dataSet=='CIFAR':
        imageArray = np.empty((1,32*32*3))
        labelArray = np.empty((1))
        return imageArray, labelArray
    else:
        imageArray = np.empty((1,28*28))
        labelArray = np.empty((1))
        return imageArray, labelArray
    
    
def getFakeData(primaryDataSet, helperDataSet, fakeClasses, instances, mmdFlag = 0, numHelperInstances=1000):
    
    '''
    Output image pixels between (-1,1)
    '''
    dataFolder = resultDir+'results/nonMMD/compressed/'+primaryDataSet
    if mmdFlag==1:
        dataFolder = resultDir+'results/crossDataSetMMDall/compressed/'+primaryDataSet
    
    fakeClasses =  sorted(fakeClasses)
    imageArray, labelArray = getInitialArray(primaryDataSet)
    
    for i in fakeClasses:
        if mmdFlag==1:
            tempImageArray = np_load(dataFolder + '/' + \
                                     primaryDataSet + '_' + helperDataSet + '_' + \
                                     str(i) + '_' + 'all' + '_' + \
                                     str(instances) + '_' + str(numHelperInstances) + \
                                     '.npy' )
            
        elif mmdFlag==0:
            tempImageArray = np_load(dataFolder + '/' + \
                                     primaryDataSet + '_' + \
                                     str(i)+'_' + \
                                     str(instances) + \
                                     '.npy')
        
        # no need to resize as model is producing image of same size
        '''
        if getChannels(dataSet)==1:
            tempImageArrayResize = np.zeros((tempImageArray.shape[0],
                                             sizeOfImage,
                                             sizeOfImage))

            for j in range(tempImageArray.shape[0]):

                tempImage = PIL.Image.fromarray(np.uint8(tempImageArray[j,:,:]))
                tempImage = tempImage.resize((sizeOfImage,sizeOfImage),
                                             PIL.Image.ANTIALIAS)
                tempImageArrayResize[j,:,:] = np.asarray(tempImage)
                
        elif getChannels(dataSet)==3:
            tempImageArrayResize = np.zeros((tempImageArray.shape[0],
                                             sizeOfImage,
                                             sizeOfImage,
                                             3))
            tempImageArray = tempImageArray.transpose([0,2,3,1])
            
            for j in range(tempImageArray.shape[0]):
                tempImage = PIL.Image.fromarray(np.uint8(tempImageArray[j,:,:,:]), 'RGB')
                tempImage = tempImage.resize((sizeOfImage,sizeOfImage),
                                             PIL.Image.ANTIALIAS)
                tempImageArrayResize[j,:,:,:] = np.asarray(tempImage) 
            tempImageArray = tempImageArray.transpose([0,3,1,2])
        '''
        
        tempImageArray = tempImageArray.reshape(tempImageArray.shape[0],-1)
        
        tempLabelArray = np.empty(tempImageArray.shape[0])
        tempLabelArray.fill(i)        
        
        #print tempImageArray.shape, imageArray.shape
        imageArray = np.concatenate([imageArray, tempImageArray])
        labelArray = np.concatenate([labelArray, tempLabelArray])
    imageArray = imageArray[1:]
    labelArray = labelArray[1:]
    
    
    # random shuffling of images and labels
    p = np.random.permutation(imageArray.shape[0])
    imageArray = imageArray[p]
    labelArray = labelArray[p]

    return imageArray, labelArray

def getRealData(dataSet, realClasses, instances, mode='train'):
    '''
    Ouput image pixels between (-1 to 1)
    ''' 
    realClasses = sorted(realClasses)
    
    numOfChannels = getChannels(dataSet)
    imageSize = getImageSize(dataSet)
    
    lenArray = len(realClasses)*instances    
    imageArray = np.zeros((lenArray, numOfChannels*imageSize*imageSize))
    labelArray = np.zeros((lenArray))
    
    initialPoint=0
    for i in realClasses:
        tupleArray = loadDataset(dataSet, i, instances, mode=mode)
        
        tempImageArray = np.asarray([ data[0].numpy().squeeze() for data in tupleArray ])
        tempLabelArray = np.asarray([ np.asarray(data[1]) for data in tupleArray ])
        
        tempImageArray = tempImageArray.reshape(tempImageArray.shape[0],-1)
        
        #print imageArray.shape, tempImageArray.shape
        imageArray[initialPoint:(initialPoint + tempImageArray.shape[0])] = tempImageArray 
        labelArray[initialPoint:(initialPoint + tempImageArray.shape[0])] = tempLabelArray 
        initialPoint = tempImageArray.shape[0] + initialPoint
        
    imageArray = imageArray[:initialPoint,:]
    labelArray = labelArray[:initialPoint]
    
    # random shuffling of images and labels
    p = np.random.permutation(imageArray.shape[0])
    imageArray = imageArray[p]
    labelArray = labelArray[p]
    
    return imageArray, labelArray

def plotConfusionMatrix(dataSet,
                        classifier,
                        classes,
                        trainSet,
                        numOfInstances,
                        cm, 
                        normalize=False,
                        numOfHelperInstances=-1,
                        title='Confusion matrix',
                        cmap='Reds'):
    """
    trainSet: can be 'Real','Fake' or 'FakeMMD'
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    # Plot normalized confusion matrix
    fig = plt.figure()
    
    if numOfHelperInstances==-1:
        fileName = dataSet+'_'+trainSet+'_'+classifier+'_'+str(numOfInstances)
    else:
        fileName = dataSet+'_'+trainSet+'_'+classifier+'_'+str(numOfInstances)+'_'+str(numOfHelperInstances)

    
    plotFolderName = 'plots/crossDataSetMMDall/cm'+'/'+dataSet+'/'
    checkAndCreateFolder(plotFolderName)
    plotFileName = plotFolderName+fileName+'.png'
        
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    
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
    
    
    plt.savefig(plotFileName, bbox_inches='tight')
    
    # change this to get correct order !!
    if showImage==0:
        plt.show()
    plt.close()
    
def plotAccuracyBar(dataSet,
                 classifier,
                 primaryInstances,
                 helperInstances,
                 accuracyArray,
                 showImage = 1):
    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    

    ind = np.arange(len(primaryInstances))
    width = 0.15
    delta = 0.02

    color = ['black', 'red', 'blue', 'green']

    histList = []
    legendList = []
    
    for i in range(accuracyArray.shape[0]):
        hist = ax.bar(ind+i*(width+delta), list(accuracyArray[:,i]), width, color=color[i])
        histList.append(hist)
        
        if i==0:
            legendList.append('Original')
        elif i==1:
            legendList.append('GAN - 0 Helper Instances')
        else:
            legendList.append('GAN - '+str(helperInstances[i-2])+' Helper Instances')
    
    print (accuracyArray)
    ax.set_xlim(-4*width,len(ind)+width)
    ax.set_ylim(0,1.5)

    ax.set_ylabel('Accuracy [out of 1]')
    ax.set_xlabel('Number of Primary Instances')
    ax.set_title(dataSet)
    

    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(primaryInstances)
    plt.setp(xtickNames, rotation=45, fontsize=10)
    
    # add to tuple
    ax.legend( tuple(histList), tuple(legendList) , loc='upper left')    

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plotFolderName = 'plots/crossDataSetMMDall/accuracy'+'/'+dataSet+'/'
    checkAndCreateFolder(plotFolderName)
    
    plotFileName = plotFolderName+dataSet+'_'+classifier+'.png'
    plt.savefig(plotFileName, bbox_inches='tight')

    plt.show()
    plt.close()

    accuracyFolderName = 'plots/crossDataSetMMDall/accuracyValues'+'/'+dataSet+'/'
    checkAndCreateFolder(accuracyFolderName)
    accuracyFileName = accuracyFolderName+dataSet+'_'+classifier+'.npy'

    # save the image in some format
    with open(accuracyFileName,'wb+') as fh:

        np_save(fh, accuracyArray, allow_pickle=False)
        sync(fh)
    
plotAccuracyBar( 'MNIST',
                 'X',
                 [100,500,1000,5000],
                 [1000,5000],
                 np.array([[0.3,0.3,0.3,0.4],[0.25,0.25,0.35,0.45],[0.27,0.27,0.37,0.47],[0.8,0.30,0,0.50]]),
                 showImage = 1)

def getPredictions(trainData, trainLabels, testData, classifier_='linear', C=10.0, degree=3, dem_red=1, scale=0):

    # scale the train and test data between 0 and 1 if required
    if scale==1:
        scaler = StandardScaler()
        scaler.fit(trainData)

        trainDataReal = scaler.transform(trainData)
        testData = scaler.transform(testData)
    
    # dimensionality reduction through PCA
    if dem_red<1:
        pca = PCA(dem_red)        
        pca.fit(trainData) 

        trainData = pca.transform(trainData)
        testData = pca.transform(testData)
    
    classifier = getClassifier(classifier_, C=C, degree=degree)
    
    start_time = time.time()
    classifier.fit(trainData, trainLabels)
    print("Time taken to classify : {} ".format((time.time() - start_time)))        
    predictions = classifier.predict(testData)  
    return predictions

# train a multi-class SVM
def classify(primaryDataSet,
             helperDataSet,
             classes_, 
             classifier_='linear', C=10.0, degree=3,
             dem_red=1, scale=0):
    
    
    class_names = getClasses(primaryDataSet)
    
    # select the classes from classes_ list
    primaryInstances = [100,500,1000,5000]
    helperInstances = [1000,5000]    
    
    # helperInstance + 2( Real + 0 ) x primaryInstances
    accuracyArray = np.zeros((len(set(primaryInstances)),len(set(helperInstances))+2))
    
    
    np.set_printoptions(precision=2)
    print ('================================')
    print ('Dataset: ' + str(primaryDataSet))
    print ('Helper Dataset: ' + str(helperDataSet))
    
    
    # get testing data
    # should do stratified or random sampling while doing this ??
    testData, testLabels = getRealData(primaryDataSet, classes_, 1000, mode='test') 
    trainDataReal, trainLabelsReal = getRealData(primaryDataSet, classes_, 5000)

    # calculate accuracy trained using original dataset
    for count,i in enumerate(primaryInstances):
        

        # get original data pertaining to classes and instances
        trainDataRealEx = trainDataReal[:i*10]
        trainLabelsRealEx = trainLabelsReal[:i*10]
        predictionsReal = getPredictions(trainDataRealEx, trainLabelsRealEx, testData, classifier_=classifier_, dem_red=dem_red, scale=scale)

        # get the classifier accuracy
        accuracyReal = sklearn.metrics.accuracy_score(testLabels, predictionsReal)
        accuracyArray[count,0]=accuracyReal
        
        # get the confusion matrix
        confusionMatrixReal = sklearn.metrics.confusion_matrix(testLabels, predictionsReal, classes_)
        plotConfusionMatrix(primaryDataSet,
                            classifier_,
                            class_names[:len(classes_)],
                            'Real',
                            i,
                            confusionMatrixReal,
                            normalize=True,
                            title='Normalized Confusion matrix')  
        
        # get GAN generated data
        trainDataGAN, trainLabelsGAN = getFakeData(primaryDataSet, helperDataSet,classes_,i,0)
        
        # augment real data with GAN generated data
        trainDataAugment = np.concatenate((trainDataRealEx,trainDataGAN),axis=0)
        trainLabelsAugment = np.concatenate((trainLabelsRealEx,trainLabelsGAN),axis=0)
        predictionsAugment = getPredictions(trainDataAugment, trainLabelsAugment, testData, classifier_=classifier_, dem_red=dem_red, scale=scale)
        
        # get the classifier accuracy
        accuracyAugment = sklearn.metrics.accuracy_score(testLabels, predictionsAugment)
        accuracyArray[count,1] = accuracyAugment
        
        # get the confusion matrix
        confusionMatrixAugment = sklearn.metrics.confusion_matrix(testLabels, predictionsAugment, classes_)
        plotConfusionMatrix(primaryDataSet,
                            classifier_,
                            class_names[:len(classes_)],
                            'Fake',
                            i,
                            confusionMatrixAugment,
                            normalize=True,
                            title='Normalized Confusion matrix')
        
        
    for count_i,i in enumerate(primaryInstances):
        for count_j,j in enumerate(helperInstances):
            if (i>j):
                continue
            # get GAN generated data ( with MMD ) pertaining to classes and instances
            # need to pass number of helper instances !!
            trainDataMMDGAN, trainLabelsMMDGAN = getFakeData(primaryDataSet,helperDataSet,classes_,i,1,j)

            # get the classifier accuracy
            # augment real data with GAN generated data
            trainDataAugment = np.concatenate((trainDataReal,trainDataMMDGAN),axis=0)
            trainLabelsAugment = np.concatenate((trainLabelsReal,trainLabelsMMDGAN),axis=0)
            predictionsAugment = getPredictions(trainDataAugment, trainLabelsAugment, testData, classifier_=classifier_, dem_red=dem_red, scale=scale)
            
            # get the classifier accuracy
            accuracyAugment = sklearn.metrics.accuracy_score(testLabels, predictionsAugment)
            accuracyArray[count_i,count_j+2]=accuracyAugment
            
            # get confusion matrix
            confusionMatrixAugment= sklearn.metrics.confusion_matrix(testLabels, predictionsAugment, classes_)
            plotConfusionMatrix(primaryDataSet,
                                classifier_,
                                class_names[:len(classes_)],
                                'FakeMMD',
                                i,
                                confusionMatrixAugment,
                                numOfHelperInstances=j,
                                normalize=True,
                                title='Normalized Confusion matrix')

    # accuracy should only be based on number of primary instances
    plotAccuracyBar(primaryDataSet,
                 classifier_,
                 primaryInstances,
                 helperInstances,
                 accuracyArray
                )
    
if __name__=='__main__':
    
    primaryDataSets = ['MNIST']
    helperDataSets = ['SVHN-BW']
    for primaryDataSet in primaryDataSets:
        for helperDataSet in helperDataSets:
            classify(primaryDataSet, 
                     helperDataSet,
                     [0,1,2,3,4,5,6,7,8,9], 
                     classifier_='sgd', C=10.0, degree=3,
                     dem_red=1, scale=0)
            
            classify(primaryDataSet, 
                     helperDataSet,
                     [0,1,2,3,4,5,6,7,8,9], 
                     classifier_='linear', C=10.0, degree=3,
                     dem_red=1, scale=0)
            
            classify(primaryDataSet, 
                     helperDataSet,
                     [0,1,2,3,4,5,6,7,8,9], 
                     classifier_='rbf', C=10.0, degree=3,
                     dem_red=1, scale=0)
            
    