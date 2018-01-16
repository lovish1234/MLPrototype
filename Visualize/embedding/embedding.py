from sklearn.datasets import fetch_mldata
import numpy as np
from numpy import load as np_load
import pandas as pd

# to parse uint format
import os
import struct

# to resize the image
import PIL
import PIL.Image

# for dimensionality reduction
from sklearn.manifold import TSNE

# for plotting the graph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox

# for shuffling the images
import random
import time

# another plotting liberary
import plotly as py
import plotly.graph_objs as go

def readIdx(filename):
    '''
    A function to read in a numpy array from MNIST/ Fashion MNIST files 
    Borrowed from Tyler Neylon

    https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    '''
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        images = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
    return images

def loadDataset(dataSet, classes, numberOfInstances, sizeOfImage, flip):
    '''
    Load a particular dataset for training/testing GAN
    64*64 size
    '''
    
    if dataSet=='MNIST':
        folder = '../../data/MNIST/raw/'
    elif dataSet=='FashionMNIST':
        folder = '../../data/FashionMNIST/raw/'
    elif dataSet=='notMNIST':
        folder = '../../data/notMNIST/raw/'

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
    imagesByClass=images[np.where(labels==classes)]
    labelsByClass = labels[np.where(labels==classes)]

    # number of images to be extracted
    imagesByClass=imagesByClass[0:numberOfInstances]
    labelsByClass=labelsByClass[0:numberOfInstances]

    imagesByClassResize = np.zeros((imagesByClass.shape[0],
                                    sizeOfImage,
                                    sizeOfImage))
    
    # resize the images to desired size
    for i in range(imagesByClass.shape[0]):
        tempImage = PIL.Image.fromarray(np.uint8(imagesByClass[i,:,:]))
        tempImage = tempImage.resize((sizeOfImage,sizeOfImage),PIL.Image.ANTIALIAS)
        imagesByClassResize[i,:,:] = np.asarray(tempImage)
    
    
    # 3D to 4D vector
    imagesByClass = np.expand_dims(imagesByClass, axis=1).astype('float32')
    #labelsByClass = np.expand_dims(labelsByClass, axis=1).astype('float32')
    imagesByClassResize = imagesByClassResize/255.0
    
    #random flipping about the y-axis
    if flip == True:
        flips = [(slice(None,None,None), slice(None,None,None),
                  slice(None,None,random.choice([-1,None]))) for _ in xrange(numberOfInstances)]
        imagesByClassResize = np.array([image[flip] for image,flip in zip(imagesByClassResize, flips)])
    
    imagesByClassResize = imagesByClassResize.reshape(imagesByClassResize.shape[0],-1)
    return imagesByClassResize, labelsByClass

def getFakeData(dataSet, fakeClasses, instances):
    '''
    64x64 images in imageArray
    '''
    fakeClasses =  sorted(fakeClasses)
    imageArray = np.empty((1,4096))
    labelArray = np.empty((1))
    for i in fakeClasses:
        tempImageArray = np_load('../../DCGAN/results/'+'compressed'+'/'+dataSet+'/'+dataSet+'_'+str(i)+'_'+str(instances)+'.npy')
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
    
    labelArray = labelArray.astype('int')
    imageArray = imageArray/255.0
    return imageArray, labelArray

def getClasses(dataSet):
    if dataSet=='MNIST':
        classNames = ['0','1','2','3','4','5','6','7','8','9']
    elif dataSet=='FashionMNIST':
        classNames = ['T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    elif dataSet=='notMNIST':
        classNames = ['A','B','C','D','E','F','G','H','I','J']
    else:
        classNames = []
    return classNames

def visualize(dataSet, classes, numOfInstances, dimension):
    '''
    For tSNE embedding of Primary Class and Generated Images
    '''
        
    realImages, realLabels = loadDataset(dataSet,classes,1000,64,False)

    # GAN generated images

    fakeImages, fakeLabels = getFakeData(dataSet,[classes],numOfInstances)
    # concatenate image pixel and labels
    fakeLabels.fill(-1)
    images = np.vstack([realImages, fakeImages])
    labels = np.hstack([realLabels, fakeLabels])

    images = np.vstack([fakeImages, realImages])
    labels = np.hstack([fakeLabels, realLabels])

    # random permutation for showing equal number of real and fake images
    #p = np.random.permutation(images.shape[0])
    #images = images[p]
    #labels = labels[p]

    featCols = [ 'pixel'+str(i) for i in range(images.shape[1]) ]
    df = pd.DataFrame(images,columns=featCols)
    df['label'] = labels

    # applying function on one of the column of dataframe
    df['label'] = df['label'].apply(lambda i: str(i))
    y = df['label'].values.astype('int')[:2001]


    # size should be number of columns+1[for index]
    print 'Size of the dataframe: {}'.format(df.shape)

    # graph details
    colors = ['r','g','b','c','m','y','k','crimson','purple','olive']
    labels = ['0','1','2','3','4','5','6','7','8','9']

    fig = plt.figure(figsize=(18,15))
    className = getClasses(dataSet)
    plotTitle = dataSet + ' ' + str(className[classes]) + ' '+str(numOfInstances)
    fig.suptitle(plotTitle, fontsize=20)

    noSNE = 2000
    time_start = time.time()

    # 2D or 3D
    if dimension==2:

        ax = fig.add_subplot(111)                
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsneResults = tsne.fit_transform(df.loc[:noSNE,featCols].values)

        print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

        dfTSNE = df.loc[:noSNE,:].copy()
        dfTSNE['x-tsne'] = tsneResults[:,0]
        dfTSNE['y-tsne'] = tsneResults[:,1]

        # distance based plotting of AnnotationBoxes
        # downsample the image for better spacing of AnnotationBoxes
        images = images.reshape(images.shape[0],64,-1)
        images = images[:,::2,::2]

        # just something big
        shown_images = np.array([[1., 1.]])  
        countReal,countFake=0,0
        for i in range(images.shape[0]):
            dist = np.sum((tsneResults[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4:
                # don't show points that are too close
                continue
            if (countReal > 45 and y[i]==classes) or (countFake> 45 and y[i]==-1):
                # don't show points for a single class beyond a threshold
                continue
            shown_images = np.r_[shown_images, [tsneResults[i]]]

            if y[i]==classes:
                colorMap = plt.get_cmap('Reds')
                countReal=countReal+1
            elif y[i]==-1:
                colorMap = plt.get_cmap('Greens')
                countFake=countFake+1
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=colorMap),
                tsneResults[i])
            # add the offsets to the visualisation
            ax.add_artist(imagebox)
            
        for i in [classes,-1]:
            tsneResultsSelect = tsneResults[np.where(y==i)]
            if i==classes:
                label='Real'
                color='r'
            else:
                label='Fake'
                color='g'
            ax.scatter(tsneResultsSelect[:,0], 
                        tsneResultsSelect[:,1], 
                        c=color, alpha=0.3, label=label)
    elif dimension==3:
        ax = fig.add_subplot(111, projection='3d')

        tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
        tsneResults = tsne.fit_transform(df.loc[:noSNE,featCols].values)

        print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

        dfTSNE = df.loc[:noSNE,:].copy()
        dfTSNE['x-tsne'] = tsneResults[:,0]
        dfTSNE['y-tsne'] = tsneResults[:,1]
        dfTSNE['z-tsne'] = tsneResults[:,2]

        #ax = fig.gca(projection='3d')

        for i in [classes,-1]:
            tsneResultsSelect = tsneResults[np.where(y==i)]
            if i==classes:
                label='Real'
                color='r'
            else:
                label='Fake'
                color='g'
            ax.scatter(tsneResultsSelect[:,0], 
                       tsneResultsSelect[:,1], 
                       tsneResultsSelect[:,2], 
                       c=color, alpha=0.3, label=label)


    ax.legend(bbox_to_anchor=(1.10, 1), 
              loc=2, borderaxespad=0)
    saveFile = 'scatter/images/'+str(dimension)+'D/'+dataSet+'/'+str(className[classes])+'_'+str(numOfInstances)+'.jpg'
    plt.savefig(saveFile, bbox_inches='tight')
    plt.show()

def getHelperClass(dataSet, primaryClass):
    '''
    Get the Helper Class for a given Primary Class
    eg. for MNIST 1, MNIST 7 is the helper class
    '''
    if dataSet=='MNIST':
        helpDict = {0:-1,1:7,2:-1,3:8,4:-1,5:6,6:5,7:1,8:3,9:-1}
    elif dataSet=='FashionMNIST':
        helpDict = {0:-1,1:-1,2:-1,3:-1,4:-1,5:-1,6:-1,7:9,8:9,9:7}
    elif dataSet=='notMNIST':
        helpDict = {0:-1,1:-1,2:-1,3:-1,4:5,5:4,6:-1,7:-1,8:-1,9:-1}
    else:
        return -1
    return helpDict[primaryClass]

# to define the function creates a scatter plot with Helper Class

if __name__=='__main__":

    datasets = ['MNIST', 'FashionMNIST', 'notMNIST']
    classes = [0,1,2,3,4,5,6,7,8,9]
    instances = [10,100,1000]
    
    for d in datasets:
        for c in classes:
            for i in instances:
                visualize(d,c,i,2)
                visualize(d,c,i,3)