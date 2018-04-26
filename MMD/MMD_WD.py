import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

# used to load and transform popular CV datasets
import torchvision
import torchvision.transforms as transforms 

#plotting the graphs
import matplotlib.pyplot as plt
from textwrap import wrap


# to save and load numpy arrays
from numpy import save as np_save
import numpy as np


#import IPython.display

# for MMD calculation
from sklearn.metrics import pairwise_kernels, pairwise_distances

# import datasets
import sys
sys.path.append('../')
from Datasets.loadDataset import loadDataset, getHelperClass, getEpochs, getChannels, getClasses, getImageSize, checkAndCreateFolder
reload(sys.modules[loadDataset.__module__])

# import configuration
import config
reload(config)
from config import *

resultDir='/Neutron3/lovish.chum/'

def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """

    # take the first m samples in RKHS as Kx and rest as Ky

    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]

    # substract the diagonal part because according to representation
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()
        
def kernelTwoSampleTest(X, Y, kernel_function='rbf', iterations=10000,
                           verbose=False, **kwargs):
    """Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.

    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    
    m = len(X)
    n = len(Y)
    
    X = X.numpy()
    X = X.reshape(X.shape[0],-1)
    Y = Y.numpy()
    Y = Y.reshape(Y.shape[0],-1)
    
    XY = np.vstack([X, Y])

    # calculate the kernel matrix given elements of both domains
    K = pairwise_kernels(XY, metric=kernel_function, **kwargs)

    mmd2u = MMD2u(K, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")

    return mmd2u

def MMDhist(dataSet, primaryClass, primaryInstance, helperInstances, batchSize):
    
    # load the datasets for which the bar graph needs to be plotted
    x = loadDataset(dataSet, [primaryClass], [primaryInstance])
    
    primaryClasses = [i for i in range(10)]
    primaryClasses = primaryClasses[:primaryClass]+primaryClasses[primaryClass+1:]
    primaryInstances = [ primaryInstance for i in range(10) ]
    
    y = loadDataset(dataSet, primaryClasses, primaryInstances )
    
    primaryTrainLoader = torch.utils.data.DataLoader(x,
                                               batch_size=batchSize, 
                                               shuffle=True,
                                               num_workers=4,
                                               drop_last = True)
    helperTrainLoader = torch.utils.data.DataLoader(y,
                                              batch_size=batchSize, 
                                              shuffle=True,
                                              num_workers=4,
                                              drop_last=True)

    mmdValues = []
    minMMDValue = 1.5
    maxMMDValue = -0.1
    
    minPrimaryDataInstance = torch.FloatTensor(batchSize, 
                                               getChannels(dataSet), 
                                               getImageSize(dataSet), 
                                               getImageSize(dataSet)).zero_()
    minHelperDataInstance = minPrimaryDataInstance
    maxPrimaryDataInstance = minPrimaryDataInstance
    maxHelperDataInstance = minPrimaryDataInstance
    
    
    
    for i, primaryData in enumerate(primaryTrainLoader, 0):
        primaryDataInstance, primaryDataLabel = primaryData
        for j, helperData in enumerate(helperTrainLoader, 0):
            
            helperDataInstance, helperDataLabel = helperData
            
            mmdValue = kernelTwoSampleTest(primaryDataInstance, helperDataInstance)
            
            # choosing the pair with minimum MMD
            if minMMDValue > mmdValue:
                minMMDValue = mmdValue
                minPrimaryDataInstance = primaryDataInstance
                minHelperDataInstance = helperDataInstance
                
            # choosing the pair with maximum MMD 
            if maxMMDValue < mmdValue:
                maxMMDValue = mmdValue
                maxPrimaryDataInstance = primaryDataInstance
                maxHelperDataInstance = helperDataInstance
            
            mmdValues.append (mmdValue)
            
            #mmdValues.append (1-kernelTwoSampleTest(primaryDataInstance, helperDataInstance))
            
    
    displayImage(maxPrimaryDataInstance, maxHelperDataInstance, maxMMDValue, dataSet, primaryClass, primaryInstance, helperInstances, 'max')
    displayImage(minPrimaryDataInstance, minHelperDataInstance, minMMDValue, dataSet, primaryClass, primaryInstance, helperInstances,'min')
    
    mmdValues = np.asarray(mmdValues)
    plt.figure()
    plt.hist(mmdValues, ec='k')

    classes = getClasses(dataSet)
    
    plt.plot()
    plt.xlabel('$MMD^2$ between batch of Primary Class and Helper Class')
    plt.ylabel('Number of Batch Pairs have $MMD^2$ value in that range')
    # "\n".join(wrap())
    plt.title(' Dataset: {} \n Primary Class: {}  \n   Primary Instances:{} \n  Helper Instances:{} \n Batch Size:{}'.
             format(dataSet,
                    classes[primaryClass], 
                    primaryInstance, 
                    helperInstances, batchSize))
    
    saveFolder= resultDir+'mmdValues'+'/'+'hist'+'/'+dataSet+'/'
    checkAndCreateFolder(saveFolder)
    saveFile = saveFolder +dataSet+'_'+str(primaryClass)+'_'+str(primaryInstance)+'_'+str(helperInstances)+'_'+str(batchSize)+'.png'
    
    plt.savefig(saveFile, bbox_inches='tight')
    plt.show()

def MMDhist_avg(dataSet, primaryClass, primaryInstances, helperInstances, batchSize):
    
    # load the datasets for which the bar graph needs to be plotted
    x = loadDataset(dataSet, [primaryClass], [primaryInstances])
    classNames = getClasses(dataSet)
    avgValues = []
    
    # check the average discrepancy between two datasets
    for m in range(10):
        
        y = loadDataset(dataSet, [m], [helperInstances])

        primaryTrainLoader = torch.utils.data.DataLoader(x,
                                                   batch_size=batchSize, 
                                                   shuffle=True,
                                                   num_workers=4,
                                                   drop_last = True)
        helperTrainLoader = torch.utils.data.DataLoader(y,
                                                  batch_size=batchSize, 
                                                  shuffle=True,
                                                  num_workers=4,
                                                  drop_last=True)

        mmdValues = []

        for i, primaryData in enumerate(primaryTrainLoader, 0):
            primaryDataInstance, primaryDataLabel = primaryData
            for j, helperData in enumerate(helperTrainLoader, 0):
                helperDataInstance, helperDataLabel = helperData

                mmdValue = kernelTwoSampleTest(primaryDataInstance, helperDataInstance)
                mmdValues.append (mmdValue)

        mmdValues = np.asarray(mmdValues)
        avgValue = np.mean(mmdValues)
        
        print dataSet, classNames[primaryClass], classNames[m], avgValue
        print 'Average Discrepancy for '+dataSet+ ' Primary Class: '+ str(classNames[primaryClass]) + ' Helper Class: '+ str(classNames[m]) + ' is :'+ str(avgValue)
        
        avgValues.append(avgValue)
        
        
    # defining plot attributes
    MMDBar(dataSet, avgValues, primaryClass, primaryInstances, helperInstances, batchSize)
    
    
def displayImage(primaryArray, helperArray, mmdScore, dataSet, primaryClass, primaryInstances, helperInstances, mode):
    
    '''
    Image comparing max and min MMD batches
    '''
    fig = plt.figure()
    
    a=fig.add_subplot(1,2,1)
    a.set_title('$X_{batch}$')

    genImage = torchvision.utils.make_grid(primaryArray[:49], nrow=7, normalize=True)
    genImage = genImage.permute(1,2,0)
    genImage = genImage.numpy()
    plt.axis('off')
    plt.imshow(genImage)
    
    a=fig.add_subplot(1,2,2)
    a.set_title('$Y_{batch}$')

    genImage = torchvision.utils.make_grid(helperArray[:49], nrow=7, normalize=True)
    genImage = genImage.permute(1,2,0)
    genImage = genImage.numpy()
    plt.axis('off')
    plt.imshow(genImage)
    
    plt.text(-100,280,'$MMD^{2}(X_{batch},Y_{batch})$: '+str(round(mmdScore,3)))
    
    plotFolderName = resultDir+'mmdValues'+'/'+mode+'/'+dataSet+'/'
    checkAndCreateFolder(plotFolderName)

    plotFileName = plotFolderName+dataSet+'_'+str(primaryClass)+'_'+str(primaryInstances)+'_'+str(helperInstances)+'_'+str(batchSize)+'.png'
    plt.savefig(plotFileName, bbox_inches='tight')
    plt.show()
    
def getMMDHist():

    primaryDataSets = [ 'CIFAR' ]
    primaryClasses = [5,6]
    primaryInstances = [5000]
    helperInstances = [5000]
    batchSizes = [128]
    
    for dataSet in primaryDataSets:
        for cls in primaryClasses:
            for instance in primaryInstances:
                for helperInstance in helperInstances:
                    
                    if instance > helperInstance:
                        continue
                    for batchSize in batchSizes:
                        MMDhist(dataSet, cls, instance, helperInstance, batchSize)

def MMDBar(dataSet, yReal, primaryClass, primaryInstances, helperInstances, batchSize):
    '''
    Plot Avg. MMD value within a dataset as a histogram
    '''   
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.ylabel('Avg. MMD Value')
    plt.xlabel('Class')

    yReal = np.asarray(yReal)
    yReal[yReal<0]=0
    yAbove = np.max(yReal)+(np.max(yReal)/3)
    yStep = (yAbove - np.min(yReal))/10
    plt.yticks(np.arange(0.0,yAbove,yStep))
    
    classNames = getClasses(dataSet)
    plt.title(dataSet+' - '+str(classNames[primaryClass]))

    # plot accuracy
    xReal = getClasses(dataSet)


    ind = np.arange(0, 10)
    ax.set_xticks(ind)
    ax.set_xticklabels(xReal)


    plt.xticks(rotation=45)
    plt.bar(ind,yReal, 0.50)
    
    plotFolderName = resultDir+'mmdValues'+'/'+dataSet+'/'
    checkAndCreateFolder(plotFolderName)

    plotFileName = plotFolderName+dataSet+'_'+str(primaryClass)+'_'+str(primaryInstances)+'_'+str(helperInstances)+'_'+str(batchSize)+'.png'
    plt.savefig(plotFileName, bbox_inches='tight')
    plt.show()
    
    
if __name__=='__main__':
    
    # get max/min and distribution of MMD values
    getMMDHist()
    
    # get average MMD values withing the dataset classes
    primaryDataSets = ['MNIST','FashionMNIST','CIFAR', 'SVHN']
    primaryClasses = [0,1,2,3,4,5,6,7,8,9]
    primaryInstances = [1000]
    helperInstances = [1000]

    for dataSet in primaryDataSets:
        for cls in primaryClasses:
            for instance in primaryInstances:
                for helperInstance in helperInstances:

                    # if the number of primary instances are larger than the number of helper
                    # instances, no need to calculate MMD
                    if instance > helperInstance:
                        continue
                    MMDhist_avg(dataSet, cls, instance, helperInstance, 50)