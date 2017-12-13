import numpy as np
import os
import struct

import torch
import torch.nn as nn

# activation functions such as sigmoid/relu
import torch.nn.functional as F

# different optimisation routines such as Adam/Adagrad
import torch.optim as optim

# automatic differenciation of variables
from torch.autograd import Variable


#plotting the graphs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# generator and discriminator models
from model import Generator, Discriminator

# to save and load numpy arrays
from os import fsync
from numpy import save as np_save
import numpy as np


# to parse uint format
import os
import struct
import argparse

# to resize the image
import PIL
import PIL.Image
from cStringIO import StringIO


parser = argparse.ArgumentParser()

# resource information
parser.add_argument('--cuda', required=False, default=True, type=bool,
help='gpu availability')
parser.add_argument('--numOfGPU', required=False, default=1, type=int, help='number of available GPU')

parser.add_argument('--imageSize', required=False, default=28, type=int,
help='image size')
parser.add_argument('--batchSize', type=int, default=1, help='train batch size')


# number of channels in input/output eg. Greyscale,1 or RGB,3 or RGBD,4
parser.add_argument('--numInputChannels', type=int, default=10, help='number of input channels')
parser.add_argument('--numOutputChannels', type=int, default=1, help='number of output channels')
#
parser.add_argument('--numGenFilter', required=False, type=int, default=128,
help='number of filters in frist layer of generator')
parser.add_argument('--numDiscFilter',required=False, type=int, default=128)

# iterations of generator and discriminator per epoch
parser.add_argument('--genIter', required=False, type=int, default=1)
parser.add_argument('--discIter',required=False, type=int, default=1)

parser.add_argument('--epochs', type=int, default=1001, help='number of train epochs')

# adam optimiser parameters
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--learningRate', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--showImage', type=float, default=0, help='plot the figure')

params = parser.parse_args()

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
    '''
    print ('Loading dataset: '+dataSet+'class: '+str(classes))
    
    if dataSet=='MNIST':
        folder = '../data/MNIST/raw/'
    elif dataSet=='FashionMNIST':
        folder = '../data/FashionMNIST/raw/'
    elif dataSet=='notMNIST':
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
    labelsByClass = np.expand_dims(labelsByClass, axis=1).astype('float32')
    imagesByClassResize = imagesByClassResize/255.0
    
    #random flipping about the y-axis
    if flip == True:
        flips = [(slice(None,None,None), slice(None,None,None),
                  slice(None,None,random.choice([-1,None]))) for _ in xrange(numberOfInstances)]
        imagesByClassResize = np.array([image[flip] for image,flip in zip(imagesByClassResize, flips)])

    
    imagesByClassResize = torch.from_numpy(imagesByClassResize)
    labelsByClass = torch.from_numpy(labelsByClass)

    dataImage = torch.utils.data.TensorDataset(imagesByClassResize,
                                          labelsByClass)
    trainLoader = torch.utils.data.DataLoader(dataImage,
                                              batch_size = batchSize,
                                              shuffle = True,
                                              num_workers = 2)
    fileName = dataSet+'_'+str(classes)+'_'+str(numberOfInstances)
    
    return trainLoader, fileName

def weight_initialisation(objectInstance):
    className = objectInstance.__class__.__name__
    if className.find('Conv') !=-1:
        # mean and std deviation for conv/deconv layers
        objectInstance.weight.data.normal_(0.0,0.02)
    elif className.find('BatchNorm') !=-1:
        objectInstance.weight.data.normal_(1.0,0.02)
        objectInstance.bias.data.fill_(0)

def train(fileName, trainLoader, instances, learningRate = 0.0002,
        optimBetas=(0.5, 0.999), epochs=5):
    '''
    Training for Deep Convolutional Generative Adversatial Network
    '''

    G = Generator(params.numInputChannels, params.numGenFilter,
            params.numOutputChannels)
    D = Discriminator(params.numOutputChannels,params.numDiscFilter)

    lossFunction = nn.BCELoss()

    genOptimiser = optim.Adam(G.parameters(),
                              lr=learningRate,
                              betas = (0.5, 0.999))
    disOptimiser = optim.Adam(D.parameters(),
                              lr=learningRate,
                              betas = (0.5, 0.999))

    discRealInput = torch.FloatTensor(params.batchSize,
                                      1,
                                      64,
                                      64)
    discFakeInput = torch.FloatTensor(params.batchSize,
                                      params.numInputChannels,
                                      1,
                                      1)
    fixedNoise = torch.FloatTensor(25,
                                   params.numInputChannels,
                                   1,
                                   1)
    fixedNoise.normal_(0,1)

    discRealLabel = torch.FloatTensor(params.batchSize)
    discRealLabel.fill_(1)

    discFakeLabel = torch.FloatTensor(params.batchSize)
    discFakeLabel.fill_(0)

    plt.figure()

    if cuda:
        G = G.cuda()
        D = D.cuda()

        lossFunction = lossFunction.cuda()

        discRealInput = discRealInput.cuda()
        discFakeInput = discFakeInput.cuda()

        discRealLabel = discRealLabel.cuda()
        discFakeLabel = discFakeLabel.cuda()

        fixedNoise = fixedNoise.cuda()

    fixedNoiseVariable = Variable(fixedNoise)
    
    
    if instances==10:
        epochs = 1001
        saveEpochs = [500,1000]
    elif instances==100:
        epochs = 501
        saveEpochs = [200,500]
    elif instances==1000:
        epochs = 101
        saveEpochs = [20,50,100]

    for epoch in range(epochs):
        for i, data in enumerate(trainLoader, 0):
            if i>10000:
                print ("Done 2000 Iterations")
                break

            # train discriminator on real data
            D.zero_grad()
            dataInstance, dataLabel = data

            if cuda:
                dataInstance = dataInstance.cuda()

            discRealInput.copy_(dataInstance)
            discRealInputVariable = Variable(discRealInput)
            discRealLabelVariable = Variable(discRealLabel)

            discRealOutput = D(discRealInputVariable)
            lossRealDisc = lossFunction(discRealOutput,
                                        discRealLabelVariable)
            lossRealDisc.backward()

            # train discriminator on fake data
            discFakeInput.normal_(0,1)
            discFakeInputVariable = Variable(discFakeInput)
            discFakeInputGen = G(discFakeInputVariable)

            discFakeLabelVariable = Variable(discFakeLabel)

            discFakeOutput = D(discFakeInputGen.detach())
            lossFakeDisc = lossFunction(discFakeOutput,
                                        discFakeLabelVariable)
            lossFakeDisc.backward()

            disOptimiser.step()

            # train generator based on discriminator
            G.zero_grad()

            genInputVariable = discFakeInputGen
            genOutputDisc = D(genInputVariable)

            lossGen = lossFunction(genOutputDisc,
                                   discRealLabelVariable)

            lossGen.backward()
            genOptimiser.step()

            if i==((instances/batchSize)-1) and epoch in saveEpochs :

                #print ('Completed processing '+str(instances)+'for'+str(saveEpoch)+'epochs.')

                # name for model and plot file
                folder = fileName.split('_')[0]
                modelFileName = 'models'+'/'+folder+'/'+fileName+'_'+str(epoch)+'.pt'
                plotFileName = 'plots'+'/'+folder+'/'+fileName+'_'+str(epoch)+'.png'

                # save the model parameters in a file
                torch.save(G.state_dict(), modelFileName)

                
                # generate samples from trained generator
                genImage = G(fixedNoiseVariable)
                genImage = genImage.data
                genImage = genImage.cpu()
                genImage = torchvision.utils.make_grid(genImage, nrow=5)
                genImage = genImage.permute(1,2,0)
                genImage = genImage.numpy()
                
                #print genImage.shape
                # plot the figure of generated samples and save
                fig = plt.figure()
                plt.imshow(genImage, cmap='gray')
                plt.axis('off')
                
                txt = 'Epoch: '+ str(epoch)
                fig.text(.45,.05,txt)
                if params.showImage==1:
                    plt.show()
                
                plt.savefig(plotFileName, bbox_inches='tight')
                plt.close('all')

def trainSamples(dataSets, classes, instances):
    for dataSet in dataSets:
        for cls in classes :
            for instance in instances:
                trainLoader, fileName = loadDataset( dataSet, cls, instance, 64, False)
                train(fileName, trainLoader, instance, 
                      params.learningRate, params.optimBetas, params.epochs)
                
if __name__=='__main__':
    dataSets = ['MNIST']
    classes = [0]
    instances = [10]
    trainSamples(dataSets, classes, instances)
