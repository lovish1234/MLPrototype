# tensor computations
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

# used to load and transform popular CV datasets
import torchvision

#plotting the graphs
import matplotlib.pyplot as plt


# to save and load numpy arrays
from numpy import save as np_save
import numpy as np


# import datasets
import sys
sys.path.append('../')
from Datasets.loadDataset import loadDataset, getTransforms, getChannels, getEpochs, getDataSetMoments, normalizeImage, unnormalizeImage, checkAndCreateFolder, getImageSamples, getAttributes, toVariable
reload(sys.modules[loadDataset.__module__])

# import configuration options
import config
reload(config)
from config import *


#%matplotlib inline
#%matplotlib notebook

import time
import imageio

# look for alternative to this solution

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

def showTrainHist(trainHist, fileName, epoch):
    
    '''
    Plot Generator and Discriminator loss function
    '''
    x = range(len(trainHist['discLoss']))

    y1 = trainHist['discLoss']
    y2 = trainHist['genLoss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    
    folder = fileName.split('_')[0]
    
    lossFolderName = resultDir+'loss/nonMMD'+'/'+folder+'/'
    checkAndCreateFolder(lossFolderName)
    lossFileName = lossFolderName+fileName+'_'+str(epoch)+'.png'
    plt.savefig(lossFileName, bbox_inches='tight')

    plt.show()
    
def getProbDist(primaryInstanceList):
    '''
    probability distribution over primaryInstances
    '''
    probList = []
    sumList = float(sum(primaryInstanceList))
    for i, item in enumerate(primaryInstanceList):
        probList.append(item/sumList)
    return probList

def train(fileName, 
          trainLoader,
          primaryInstanceList, 
          numClasses,
          numOutputChannels = 1, 
          learningRate = 0.0002,
          optimBetas=(0.5, 0.999), epochs=5):
    '''
    Training for Deep Convolutional Generative Adversatial Network
    '''
    folder = fileName.split('_')[0]
    instances = sum(primaryInstanceList)

    # generator takes input channels, number of labels, number of generative filters, number of output channels
    G = Generator(numInputChannels, 
                  numClasses,
                  numGenFilter,
                  numOutputChannels)
    D = Discriminator(numOutputChannels,
                      numClasses,
                      numDiscFilter)
    
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    
    lossFunction = nn.BCELoss()

    genOptimiser = optim.Adam(G.parameters(),
                              lr=learningRate,
                              betas = optimBetas)
    disOptimiser = optim.Adam(D.parameters(),
                              lr=learningRate,
                              betas = optimBetas)
    
    numElementsNeededPerClass = 10
    
    fixedNoise = torch.randn(numElementsNeededPerClass*numClasses,
                             numInputChannels,
                             1,
                             1)
    
    # class from which the GAN should output a distribution
    fixedNoiseClass = torch.zeros(numElementsNeededPerClass*numClasses,
                                  numClasses,
                                  1,
                                  1)
    
    classIndex = torch.zeros(numElementsNeededPerClass,1)
    for i in range(numClasses-1):
        temp = torch.ones(numElementsNeededPerClass,1)+i
        classIndex = torch.cat([classIndex, temp], 0)
    
    fixedNoiseClass = fixedNoiseClass.squeeze().scatter_(1, classIndex.type(torch.LongTensor), 1)
    fixedNoiseClass = fixedNoiseClass.view(-1,
                                           numClasses,
                                           1,
                                           1)
    
    

    # added imagesize
    discRealInput = torch.FloatTensor(batchSize,
                                      numOutputChannels,
                                      imageSize,
                                      imageSize)
    
    discRealInputClass = torch.zeros(batchSize,
                                          numClasses,
                                          imageSize,
                                          imageSize)
    
    discFakeInput = torch.FloatTensor(batchSize,
                                      numInputChannels,
                                      1,
                                      1)
    
    discFakeInputClass = torch.zeros(batchSize,
                                      numClasses,
                                      1,
                                      1)
    
    discRealLabel = torch.FloatTensor(batchSize)
    discRealLabel.fill_(1)

    discFakeLabel = torch.FloatTensor(batchSize)
    discFakeLabel.fill_(0)
    
    if instances < batchSize:
 
        discRealInput = torch.FloatTensor(instances,
                                      numOutputChannels,
                                      imageSize,
                                      imageSize)
        discFakeInput = torch.FloatTensor(instances,
                                      numInputChannels,
                                      1,
                                      1)
        
        discRealInputClass = torch.zeros(instances,
                                      numClasses,
                                      imageSize,
                                      imageSize)
        discFakeInputClass = torch.zeros(instances,
                                          numClasses,
                                          1,
                                          1)
        
        discRealLabel = torch.FloatTensor(instances)
        discRealLabel.fill_(1)

        discFakeLabel = torch.FloatTensor(instances)
        discFakeLabel.fill_(0)


    if cuda:
        G = G.cuda()
        D = D.cuda()

        lossFunction = lossFunction.cuda()

        discRealInput = discRealInput.cuda()
        discFakeInput = discFakeInput.cuda()
        
        discRealInputClass = discRealInputClass.cuda()
        discFakeInputClass = discFakeInputClass.cuda()

        discRealLabel = discRealLabel.cuda()
        discFakeLabel = discFakeLabel.cuda()

        fixedNoise = fixedNoise.cuda()
        fixedNoiseClass = fixedNoiseClass.cuda()
        

    fixedNoiseVariable = Variable(fixedNoise)
    fixedNoiseClassVariable = Variable(fixedNoiseClass)
    
    # can take the oneHot representation to feed into generator directly from here
    oneHotGen = torch.zeros(numClasses, numClasses)
    oneHotGen = oneHotGen.scatter_(1, torch.LongTensor([i for i in range(numClasses)]).view(numClasses,1), 1).view(numClasses, numClasses, 1, 1)
    
    # can take the oneHot representation to feed into discriminator directly from here
    oneHotDisc = torch.zeros([numClasses, numClasses, imageSize, imageSize])
    for i in range(numClasses):
        oneHotDisc[i,i,:,:] = 1
    
    
    trainHist = {}
    trainHist['discLoss'] = [] 
    trainHist['genLoss'] = []
    trainHist['perEpochTime'] = []
    trainHist['totalTime'] = []
    
    
    imageList = []
    
    for epoch in range(epochs):
        
        generatorLosses = []
        discriminatorLosses = []
        
        epochStartTime = time.time()
        
        for i, data in enumerate(trainLoader, 0):

            # train discriminator on real data,
            D.zero_grad()
            dataInstance, dataClass = data
            
            
            # one-hot encoding for discriminator class input
            dataClass = oneHotDisc[dataClass]

            if cuda:
                dataInstance = dataInstance.cuda()
                dataClass = dataClass.cuda()
            
            discRealInput.copy_(dataInstance)   
            discRealInputClass.copy_(dataClass)
            
            
            discRealInputVariable = Variable(discRealInput)
            discRealInputClassVariable = Variable(discRealInputClass)
            discRealLabelVariable = Variable(discRealLabel)
            
            discRealOutput = D(discRealInputVariable, 
                               discRealInputClassVariable)
            
            lossRealDisc = lossFunction(discRealOutput,
                                        discRealLabelVariable)
            lossRealDisc.backward()

            # train discriminator on fake data
            discFakeInput.normal_(0,1)
            
            # change this as instances x numClasses x imagesize x imagesize
            if instances < batchSize:
                #dataFakeClass = (torch.rand(instances)*numClasses).type(torch.LongTensor)
                dataFakeClass = torch.from_numpy(np.random.choice(numClasses, instances, p=getProbDist(primaryInstanceList)))
            else :
                #dataFakeClass = (torch.rand(batchSize)*numClasses).type(torch.LongTensor)
                dataFakeClass = torch.from_numpy(np.random.choice(numClasses, batchSize, p=getProbDist(primaryInstanceList)))

            
            discFakeInputClass = oneHotDisc[dataFakeClass]
            genFakeInputClass = oneHotGen[dataFakeClass]
            
            if cuda:
                discFakeInputClass = discFakeInputClass.cuda()
                genFakeInputClass = genFakeInputClass.cuda()
            
            discFakeInputVariable = Variable(discFakeInput)
            discFakeInputClassVariable = Variable(discFakeInputClass)
            genFakeInputClassVariable = Variable(genFakeInputClass)
            discFakeLabelVariable = Variable(discFakeLabel)
            
            
            
            discFakeInputGen = G(discFakeInputVariable, 
                                 genFakeInputClassVariable)

            # change the gradients of discriminator only
            discFakeOutput = D(discFakeInputGen.detach(), 
                               discFakeInputClassVariable)
            
            lossFakeDisc = lossFunction(discFakeOutput,
                                        discFakeLabelVariable)
            lossFakeDisc.backward()

            disOptimiser.step()
            
            # log the loss for discriminator
            discriminatorLosses.append((lossRealDisc+lossFakeDisc).data[0])
            
            # train generator based on discriminator
            G.zero_grad()

            genInputVariable = discFakeInputGen
            
            # get the class function here
            genOutputDisc = D(genInputVariable, 
                              discFakeInputClassVariable)

            lossGen = lossFunction(genOutputDisc,
                                   discRealLabelVariable)

            lossGen.backward()
            genOptimiser.step()
            
            # log the loss for generator
            generatorLosses.append(lossGen.data[0])
          
        # create an image for every epoch
        # generate samples from trained generator
        genImage = G(fixedNoiseVariable, fixedNoiseClassVariable)
        genImage = genImage.data
        genImage = genImage.cpu()

        genImage = torchvision.utils.make_grid(genImage, nrow=10)
        genImage = (genImage/2) + 0.5
        genImage = genImage.permute(1,2,0)
        genImage = genImage.numpy()
        
        plt.figure()
        fig = plt.figure(figsize=(20,10))
        plt.imshow(genImage)
        plt.axis('off')

        txt = 'Epoch: '+ str(epoch+1)
        fig.text(.45,.05,txt)


        plt.savefig('y.png', bbox_inches='tight')
            
        imageList.append(imageio.imread('y.png'))
        
        epochEndTime = time.time()
        perEpochTime = epochEndTime - epochStartTime 
        discLoss = torch.mean(torch.FloatTensor(discriminatorLosses))
        genLoss = torch.mean(torch.FloatTensor(generatorLosses))        
        print ('Epoch : [%d/%d] time: %.2f, loss_d: %.3f, loss_g: %.3f'% (epoch+1,
                                                                 epochs,
                                                                 perEpochTime,
                                                                 discLoss,
                                                                 genLoss))
    
        if epoch==(epochs-1) :
            print ('Completed processing '+str(instances)+'for '+str(epochs)+'epochs.')

            plotFolderName = resultDir+'plots/nonMMD'+'/'+folder+'/'
            checkAndCreateFolder(plotFolderName)
            plotFileName = plotFolderName+fileName+'_'+str(epochs)+'.png'

            plt.imshow(genImage)
            plt.savefig(plotFileName, bbox_inches='tight')
            plt.close('all')
            
            # create gif animation
            animFolderName = resultDir+'animation/nonMMD'+'/'+folder+'/'
            checkAndCreateFolder(animFolderName)
            animFileName = animFolderName+fileName+'_'+str(epochs)+'.gif'
            
            imageio.mimsave(animFileName, imageList, fps=5)
        

        trainHist['discLoss'].append(discLoss)
        trainHist['genLoss'].append(genLoss)
        

    # save the model parameters in a file
    modelFolderName = resultDir+'models/nonMMD'+'/'+folder+'/'
    checkAndCreateFolder(modelFolderName)
    modelFileName = modelFolderName+fileName+'_'+str(epochs)+'.pt'
    torch.save(G.state_dict(), modelFileName)
    
    showTrainHist(trainHist, fileName, epoch)
    
def trainSamples(primaryDomain, 
                 helperDomain, 
                 primaryInstances, 
                 helperInstances, 
                 fewShotInstances, 
                 primaryClasses, 
                 helperClasses, 
                 fewShotClasses,
                 epochs=100):
    
        primaryClassList, primaryInstanceList, helperClassList, helperInstanceList = getAttributes(primaryInstances, 
                                                                                         helperInstances,
                                                                                         fewShotInstances, 
                                                                                         primaryClasses, 
                                                                                         helperClasses, 
                                                                                         fewShotClasses)   
        
        # find the k-NN between train and test images
        x = loadDataset(primaryDomain, primaryClassList, primaryInstanceList, mode='train')
        numOutputChannels = getChannels(primaryDomain)

        
        tempImageArray = np.asarray([ data[0].numpy().squeeze() for data in x ])
        tempLabelArray = np.asarray([ np.asarray(data[1]) for data in x ])
        tempImageArray = tempImageArray.reshape(tempImageArray.shape[0],-1)

        sampleImage = getImageSamples(primaryDomain, tempImageArray, numOfChannels=numOutputChannels)

        # plot the figure of generated samples and save
        fig = plt.figure(figsize=(20,10))
        plt.imshow(sampleImage)
        plt.axis('off')  
        plt.show()

        # load the dataset parallaly 
        primaryTrainLoader=torch.utils.data.DataLoader(x,
                    batch_size=batchSize, shuffle=True,
                    num_workers=4, drop_last=True)
        
        fileName = primaryDomain+'_'+str(fewShotClasses)+'_'+str(primaryInstances)+'_'+str(fewShotInstances)
        train(fileName, 
              primaryTrainLoader, 
              primaryInstanceList, 
              primaryClasses, 
              numOutputChannels=numOutputChannels, 
              epochs=epochs)
        
def run (pD, pI, fsI, fsC, e):
    
    #defining the domain
    primaryDomain = pD
    helperDomain = 'SVHN'
    selectModel(primaryDomain)
    
    # primaryDomain has plenty of data
    primaryInstances = pI
    fewShotInstances = fsI
    fewShotClasses = fsC
    
    # helperDomain has scarcity of data for few classes
    helperInstances = 5000
    
    # number of classes in primary and helper domain
    primaryClasses = 10
    helperClasses = 10
    
    
    trainSamples(primaryDomain, 
                 helperDomain, 
                 primaryInstances, 
                 helperInstances, 
                 fewShotInstances, 
                 primaryClasses, 
                 helperClasses, 
                 fewShotClasses,
                 epochs =e)

if __name__=='__main__':
    
    # primaryDomain, instances, few shot instances, few shot classes, epochs
    run ('MNIST', 5000, 1000, [0], 30)
