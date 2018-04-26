import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

# used to load and transform popular CV datasets
import torchvision
import torchvision.transforms as transforms 

#plotting the graphs
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# format text within a plot
from textwrap import wrap


# to save and load numpy arrays
from numpy import save as np_save
import numpy as np


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
    
def MMDhist(primaryDomain,
            helperDomain,
            primaryClass,
            helperClassList,
            primaryInstance,
            helperInstance,
            batchSize):
    
    # load the datasets for which the bar graph needs to be plotted
    x = loadDataset(primaryDomain, [primaryClass], [primaryInstance], mode='train')
    primaryTrainLoader = torch.utils.data.DataLoader(x,
                                               batch_size=batchSize, 
                                               shuffle=True,
                                               num_workers=4,
                                               drop_last = True)
    
    avgMMDValues = []
    for helperClass in helperClassList :
        
        y = loadDataset(helperDomain, [helperClass], [helperInstance], mode='train')    

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
                mmdValues.append(mmdValue)
        
        mmdValues = np.asarray(mmdValues)
        avgMMDValues.append(np.mean(mmdValues))
    

    # plot the average MMD Values
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.ylabel('Avg. MMD Value')
    plt.xlabel('Class')
    
    avgMMDValues = np.asarray(avgMMDValues)
    avgMMDValues[avgMMDValues<0] = 0
    
    # plot adjustments
    yAbove = np.max(avgMMDValues)+(np.max(avgMMDValues)/3)
    yStep = (np.max(avgMMDValues))/10
    
    plt.yticks(np.arange(0.0,yAbove,yStep))
    
    classNames = getClasses(primaryDomain)
    plt.title(primaryDomain+' - '+str(classNames[primaryClass]))
    
    # x-axis adjustments
    xReal = getClasses(helperDomain)
    ind = np.arange(0, len(xReal))
    ax.set_xticks(ind)
    ax.set_xticklabels(xReal)
    
    plt.xticks(rotation=45)
    plt.bar(ind, avgMMDValues, 0.50)   
    
    plotFolderName = resultDir+'mmdValues'+'/'+primaryDomain+'/'
    checkAndCreateFolder(plotFolderName)

    plotFileName = plotFolderName+primaryDomain+'_'+str(primaryClass)+'_'+str(helperDomain)+'_'+str(batchSize)+'.png'
    plt.savefig(plotFileName, bbox_inches='tight')
    plt.show()
    

def displayImage(primaryArray, helperArray, mmdScore, dataSet, primaryClass, primaryInstances, helperInstances, mode):
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

    plotFileName = \ plotFolderName+dataSet+'_'+str(primaryClass)+'_'+str(primaryInstances)+'_'+str(helperInstances)+'_'+str(batchSize)+'.png'
    plt.savefig(plotFileName, bbox_inches='tight')
    plt.show()
    

def getMMDHist(pd, hd, p):
    
    primaryDomain = pd
    helperDomain = hd
    
    # single primary class and a list of helper classes
    # to compare with
    primaryClass = p
    helperClassList = [0,1,2,3,4,5,6,7,8,9]
    
    primaryInstanceList = 5000
    helperInstanceList = 5000
    

    MMDhist(primaryDomain,
            helperDomain,
            primaryClass,
            helperClassList,
            primaryInstanceList,
            helperInstanceList,
            batchSize)

if __name__=='__main__':
    
    primaryDomain = ['MNIST']
    helperDomain = ['USPS']
    primaryClassList = [0,1,2,3,4,5,6,7,8,9]
    for i in range(len(primaryDomain)):
        for p in primaryClassList:
            getMMDHist(primaryDomain[i], helperDomain[i], p)
            
    primaryDomain = ['USPS']
    helperDomain = ['MNIST']
    primaryClassList = [0,1,2,3,4,5,6,7,8,9]
    for i in range(len(primaryDomain)):
        for p in primaryClassList:
            getMMDHist(primaryDomain[i], helperDomain[i], p)