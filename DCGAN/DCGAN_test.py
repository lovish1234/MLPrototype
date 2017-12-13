import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# used to load and transform popular CV datasets
import torchvision
import torchvision.transforms as transforms 

from model import Generator, Discriminator

# to save and load numpy arrays
from os import fsync
import numpy as np
from numpy import save as np_save
from numpy import load as np_load

# to parse uint format
import os
import struct
import argparse

import PIL.Image
from cStringIO import StringIO


parser = argparse.ArgumentParser()

# resource information
parser.add_argument('--cuda', required=False, default=True, type=bool, help='gpu availability')
parser.add_argument('--numOfGPU', required=False, default=1, type=int, help='number of available GPU')


# number of channels in input/output eg. Greyscale,1 or RGB,3 or RGBD,4
parser.add_argument('--numInputChannels', type=int, default=10, help='number of input channels')
parser.add_argument('--numOutputChannels', type=int, default=1, help='number of output channels')

#
parser.add_argument('--numGenFilter', required=False, type=int, default=128,
help='numberof filters in frist layer of generator')


parser.add_argument('--numOfSamples', type=int, default=1000)
parser.add_argument('--showImage', required=False, type=int, default=0, help='plot the figures')

params = parser.parse_args()

def sync(fh):
    """
    This makes sure data is written to disk, so that buffering doesn't influence the timings.
    """
    fh.flush()
    fsync(fh.fileno())
    
def getEpochs(numOfInstances):
    '''
    Inputs:
    
    numOfInstances : Int : Number of instances used from original dataset
    
    Outputs:
    
    epochs : Int : Number of epochs for which Generator has trained
    [500,1000] for 10
    [100,200,1000] for 100
    [20,50,100] for 1000
    
    '''
    if numOfInstances==10:
        return 500
    elif numOfInstances==100:
        return 200
    elif numOfInstances==1000:
        return 50

def test(dataSet, generatedClass, numOfInstances):
    '''
    Inputs :
    
    dataSets : List : Datasets for which samples are to be genrated
    instances : List : Number of instances to be used from original dataset
    classes : List : Classes for which samples are to be generated
    
    Output :
    
    File with 1000 compressed images generated by GAN
    
    '''
    
    modelFolder = 'models'+'/'+dataSet
    modelFile = modelFolder+'/'+dataSet+'_'+str(generatedClass)+'_'+ \
    str(numOfInstances)+'_'+str(getEpochs(numOfInstances))+'.pt'    
    
    print ('Generating examples for Dataset: '+dataSet+
           ' Class: '+str(generatedClass)+
           ' Instances: '+str(numOfInstances)+
           ' Epochs: '+str(getEpochs(numOfInstances)))
    
    # load the model leannt during training
    G = Generator(params.numInputChannels, params.numGenFilter, params.numOutputChannels)
    G.load_state_dict(torch.load(modelFile))

    genImageConcat = np.empty(1)
    for sample in range(params.numOfSamples):
        noise = torch.FloatTensor(1,
                                  params.numInputChannels,
                                  1,
                                  1)
        noise.normal_(0,1)

        if params.cuda:
            G = G.cuda()
            noise = noise.cuda()
        noiseVariable = Variable(noise)

        genImage = G(noiseVariable)
        genImage = genImage.data
        genImage = genImage.cpu()
        genImage = genImage.numpy()
        
        maxImage = np.max(genImage)
        minImage = np.min(genImage)
        #print maxImage, minImage
        
        genImage = np.multiply(np.divide((genImage-minImage),(maxImage-minImage)),255.0)
        maxImage = np.max(genImage)
        minImage = np.min(genImage)
        
        if sample==0:
            genImageConcat = genImage
        elif sample>=1:
            genImageConcat = np.concatenate((genImageConcat, genImage),
                                            axis=0)

    path = '../DCGAN/results'+'/'+'compressed'+'/'+dataSet+'/'+ dataSet + '_' \
            + str(generatedClass) + '_' + str(numOfInstances) + '.npy'
        
    # save the image in some format
    with open(path,'wb+') as fh:
        genImageConcat = np.squeeze(genImageConcat)
        np_save(fh, genImageConcat, allow_pickle=False)
        sync(fh)

def showImageMatrix(dataSet, cls, instances):
    '''
    Inputs :
    
    dataSets : List : Datasets for which samples are to be genrated
    instances : List : Number of instances to be used from original dataset
    classes : List : Classes for which samples are to be generated
    
    Outputs :
    
    5x5 image matrix 
    
    '''
    fileName = '../DCGAN/results/compressed'+'/'+dataSet+'/'+dataSet+'_'+str(cls)+'_'+str(instances)+'.npy'
    images = np_load(fileName)
    
    # get random list of images to be displayed
    randomList = np.random.randint(0,1000,(25))
    imageList = images[randomList]
    
    
    fmt = 'png'
    
    # need to generalise this snippet
    fig, axes = plt.subplots(5,5)
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.7, hspace=-0.1)
    plt.axis('off')
    
    for i in range(5):
        for j in range(5):
            f = StringIO()
            image = PIL.Image.fromarray(np.uint8(imageList[i*5+j]))
            image.save(f,fmt)
            axes[i,j].imshow(image, cmap='Greys_r')
            axes[i,j].axis('off')
            axes[i,j].set_xticklabels([])
            axes[i,j].set_yticklabels([])
            axes[i,j].set_aspect("equal")
    
    plotFileName = 'results'+'/'+'samples'+'/'+dataSet+'/'+dataSet+'_'+str(cls)+'_'+str(instances)+'.png'
    plt.savefig(plotFileName, bbox_inches='tight')
    if params.showImage==1:
        plt.show()
        
def generateSamples(dataSets, classes, instances):
    '''
    Inputs :
    
    dataSets : List : Datasets for which samples are to be genrated
    instances : List : Number of instances to be used from original dataset
    classes : List : Classes for which samples are to be generated
    
    Outputs :
    
    .npy files with 1000 generated samples
    '''

    for dataSet in dataSets:
        for instance in instances:
            for cls in classes:
                # take it as GAN generated class
                test(dataSet, cls, instance)
                showImageMatrix(dataSet, cls, instance)

  
if __name__=="__main__":
    dataSets = ['MNIST','FashionMNIST','notMNIST']
    instances = [10,100,1000]
    classes = [0,1,2,3,4,5,6,7,8,9]
    generateSamples(dataSets, classes, instances)
