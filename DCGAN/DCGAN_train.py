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

# plotting the loss function wrt epochs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# used to load and transform popular CV datasets
import torchvision
import torchvision.transforms as transforms 
import argparse

import random
from skimage import transform

from model import Generator, Discriminator

parser = argparse.ArgumentParser()

# resource information
parser.add_argument('--cuda', required=False, default=True, type=bool,
help='gpu availability')
parser.add_argument('--numOfGPU', required=False, default=1, type=int, help='number of available GPU')

parser.add_argument('--imageSize', required=False, default=28, type=int,
help='image size')
parser.add_argument('--batchSize', type=int, default=5, help='train batch size')


# number of channels in input/output eg. Greyscale,1 or RGB,3 or RGBD,4
parser.add_argument('--numInputChannels', type=int, default=1, help='number of input channels')
parser.add_argument('--numOutputChannels', type=int, default=1, help='number of output channels')
#
parser.add_argument('--numGenFilter', required=False, type=int, default=128,
help='numberof filters in frist layer of generator')
parser.add_argument('--numDiscFilter',required=False, type=int, default=128)

# iterations of generator and discriminator per epoch
parser.add_argument('--genIter', required=False, type=int, default=1)
parser.add_argument('--discIter',required=False, type=int, default=1)

parser.add_argument('--epochs', type=int, default=200, help='number of train epochs')

# adam optimiser parameters
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--learningRate', type=float, default=0.0002, help='learning rate for generator, default=0.0002')

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

def loadDataset(dataSet, classes, numberOfInstances, sizeOfImage, flip=True):
    '''
    Load a particular dataset for testing GAN
    '''
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

    #print("images: {}".format(images.shape))
    #print("labels: {}".format(labels.shape))

    # get images only belonging to a particular class
    imagesByClass=images[np.where(labels==classes)]
    labelsByClass = labels[np.where(labels==classes)]

    # number of images to be extracted
    imagesByClass=imagesByClass[0:numberOfInstances]
    labelsByClass=labelsByClass[0:numberOfInstances]


    # 3D to 4D vector
    imagesByClass = np.expand_dims(imagesByClass, axis=1).astype('float32')
    labelsByClass = np.expand_dims(labelsByClass, axis=1).astype('float32')
    imagesByClass/=255.0


    # resize images, code may be improved
    #print (imagesByClass.shape[0], imagesByClass.shape[1])
    imagesByClassResize = np.zeros((imagesByClass.shape[0],
                                    imagesByClass.shape[1],
                                    sizeOfImage,
                                    sizeOfImage))

    # resizing the image to desired size
    for i in range(imagesByClass.shape[0]):
        for j in range(imagesByClass.shape[1]):
            imagesByClassResize[i,j,:,:] = transform.resize(imagesByClass[i,j,:,:],
                                                            (sizeOfImage,
                                                             sizeOfImage))
    if flip == True:
    #random flipping about the y-axis
        flips = [(slice(None,None,None), slice(None,None,None),
                  slice(None,None,random.choice([-1,None]))) for _ in xrange(numberOfInstances)]
        imagesByClassResize = np.array([image[flip] for image,flip in zip(imagesByClassResize, flips)])

    imagesByClassResize = torch.from_numpy(imagesByClassResize)
    labelsByClass = torch.from_numpy(labelsByClass)

    data = torch.utils.data.TensorDataset(imagesByClassResize,
                                          labelsByClass)
    trainLoader = torch.utils.data.DataLoader(data,
                                              batch_size = 5,
                                              shuffle = True,
                                              num_workers = 2)
    fileName = dataSet+'_'+str(classes)+'_'+str(numberOfInstances)

    return trainLoader, fileName

def weight_initialisation(objectInstance):
    className = objectInstance.__class__.__name__
    #print (className)
    if className.find('Conv') !=-1:
        # mean and std deviation for conv/deconv layers
        objectInstance.weight.data.normal_(0.0,0.02)
    elif className.find('BatchNorm') !=-1:
        objectInstance.weight.data.normal_(1.0,0.02)
        objectInstance.bias.data.fill_(0)


saveEpochs = [10,20,50,100,200,500,1000]


def train(fileName, trainLoader, instances, learningRate = 0.0002,
        optimBetas=(0.5, 0.999), epochs=5):
    '''
    Training for Deep Convolutional Generative Adversatial Network
    '''

    G = Generator(params.numInputChannels, params.numGenFilter,
            params.numOutputChannels)
    D = Discriminator(params.numOutputChannels, params.numDiscFilter)

    lossFunction = nn.BCELoss()

    genOptimiser = optim.Adam(G.parameters(),
                              lr=params.learningRate,
                              betas = (0.5, 0.999))
    disOptimiser = optim.Adam(D.parameters(),
                              lr=params.learningRate,
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

    if params.cuda:
        G = G.cuda()
        D = D.cuda()

        lossFunction = lossFunction.cuda()

        discRealInput = discRealInput.cuda()
        discFakeInput = discFakeInput.cuda()

        discRealLabel = discRealLabel.cuda()
        discFakeLabel = discFakeLabel.cuda()

        fixedNoise = fixedNoise.cuda()

    fixedNoiseVariable = Variable(fixedNoise)

    for epoch in range(params.epochs):
        for i, data in enumerate(trainLoader, 0):
            if i>10000:
                print ("Done 2000 Iterations")
                break

            # train discriminator on real data
            D.zero_grad()
            dataInstance, dataLabel = data

            if params.cuda:
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

            if i==((instances/params.batchSize)-1) and epoch in saveEpochs:

                print (epoch)

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

                # plot the figure of generated samples and save
                plt.figure()
                plt.imshow(genImage)
                plt.axis('off')
                plt.savefig(plotFileName, bbox_inches='tight')
                plt.close('all')

def main():

    dataBase = ['FashionMNIST', 'MNIST', 'notMNIST']
    classes = [0,1,2,3,4,5,6,7,8,9]
    instances = [ 1000 ]

    optimBetas = (params.beta1, params.beta2)

    for x in dataBase:
        for c in classes :
            for i in instances:
                # put an option weather to flip horizontaly
                trainLoader, fileName = loadDataset(x, c, i, 64, flip=False)
                train(fileName, trainLoader, i, params.learningRate, optimBetas, params.epochs)

if __name__=='__main__':
    main()
