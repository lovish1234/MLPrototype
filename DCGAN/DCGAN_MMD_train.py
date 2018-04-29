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

from model_28 import Generator, Discriminator
numGenFilter=64
numDiscFilter=32
imageSize = 28

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

# first helper class and then primary class

def train(primaryFileName, 
          helperFileName, 
          primaryTrainLoader, 
          helperTrainLoader, 
          primaryInstances, 
          numOutputChannels=1, 
          learningRate = 0.0002,
          optimBetas=(0.5, 0.999), epochs = 5):
    '''
    Training for Deep Convolutional Generative Adversatial Network
    '''
    
    # define the model
    G = Generator(numInputChannels, 
                  numGenFilter,
                  numOutputChannels)
    D = Discriminator(numOutputChannels, 
                      numDiscFilter)
    lossFunction = nn.BCELoss()
    genOptimiser = optim.Adam(G.parameters(),
                              lr=learningRate,
                              betas = optimBetas)
    disOptimiser = optim.Adam(D.parameters(),
                              lr=learningRate,
                              betas = optimBetas)
    discRealInput = torch.FloatTensor(batchSize,
                                      numOutputChannels,
                                      imageSize,
                                      imageSize)
    discFakeInput = torch.FloatTensor(batchSize,
                                      numInputChannels,
                                      1,
                                      1)
    fixedNoise = torch.FloatTensor(25,
                                   numInputChannels,
                                   1,
                                   1)
    fixedNoise.normal_(0,1)

    discRealLabel = torch.FloatTensor(batchSize)
    discFakeLabel = torch.FloatTensor(batchSize)
    discRealLabel.fill_(1)
    discFakeLabel.fill_(0)
    
    # for processing on a GPU
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
    
    # assume that helper primaryInstances are always more than primary primaryInstances
    # define primary Epochs and helper Epochs
    primaryEpochs = epochs
    helperEpochs = 10

    # for processing on a GPU
    if cuda:

        discRealInput = discRealInput.cuda()
        discFakeInput = discFakeInput.cuda()

        discRealLabel = discRealLabel.cuda()
        discFakeLabel = discFakeLabel.cuda()
    
    
    print ("Starting training with helper class.")
    plt.figure()

    # training with helper class
    for epoch in range(helperEpochs):
        for i, primaryData in enumerate(primaryTrainLoader, 0):
            for j, helperData in enumerate(helperTrainLoader, 0):
                
                #print ('Epoch : {} Primary Class Batch : {}. Helper Class Batch : {}.'.format(epoch+1,i+1,j+1))
                
                primaryDataInstance, primaryDataLabel = primaryData
                helperDataInstance, helperDataLabel = helperData
                
                # calculate MMD between two batches of data
                mmd = (1-kernelTwoSampleTest(primaryDataInstance, helperDataInstance))
                mmd = torch.from_numpy(np.asarray([mmd]))
                mmdVariable = Variable(mmd.float().cuda())
                
                # weight given to the term 
                lambdaMMD = 1.0
                lambdaMMD = torch.from_numpy(np.asarray([lambdaMMD]))
                lambdaMMDVariable = Variable(lambdaMMD.float().cuda())
                
                D.zero_grad()
                
                # train GAN using helper data instance
                if cuda:
                    helperDataInstance = helperDataInstance.cuda()

                discRealInput.copy_(helperDataInstance)                
                discRealInputVariable = Variable(discRealInput)
                
                # should we treat this as 1 ??
                discRealLabelVariable = Variable(discRealLabel)
                discRealOutput = D(discRealInputVariable)
                lossRealDisc = lambdaMMDVariable*mmdVariable*lossFunction(discRealOutput,
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
                # for every epoch the gradients are reset to 0
                # the discriminator should start to confuse fake primaryInstances
                # with true primaryInstances

                G.zero_grad()

                genInputVariable = discFakeInputGen
                genOutputDisc = D(genInputVariable)

                lossGen = lossFunction(genOutputDisc,
                                       discRealLabelVariable)

                lossGen.backward()
                genOptimiser.step()
            
            # for every epoch show the image
            if (i==0) and epoch==(helperEpochs-1):

                #print ('Completed processing '+str(primaryInstances)+'for'+str(epoch)+'epochs.')

                # name for model and plot file
                folder, primaryClass, _ = primaryFileName.split('_')
                _, helperClass, helperInstances = helperFileName.split('_')
                
                fileName = folder + '_' + str(primaryClass) + '_' + str(helperClass) + '_' + \
                           str(primaryInstances) + '_' + str(helperInstances)
                
                # generate samples from trained generator
                genImage = G(fixedNoiseVariable)
                genImage = genImage.data
                genImage = genImage.cpu()
                genImage = torchvision.utils.make_grid(genImage, nrow=5)
                
                genImage = genImage/2 + 0.5
                genImage = genImage.permute(1,2,0)
                genImage = genImage.numpy()
                
                #print genImage.shape
                # plot the figure of generated samples and save
                fig = plt.figure()
                plt.imshow(genImage, cmap='gray')
                plt.axis('off')
    
    if primaryInstances < batchSize:
        discRealInput = torch.FloatTensor(primaryInstances,
                                          numOutputChannels,
                                          imageSize,
                                          imageSize)
        # why only one as width and height ? Passing through generator.
        discFakeInput = torch.FloatTensor(primaryInstances,
                                          numInputChannels,
                                          1,
                                          1)
        discRealLabel = torch.FloatTensor(primaryInstances)
        discFakeLabel = torch.FloatTensor(primaryInstances)
        discRealLabel.fill_(1)
        discFakeLabel.fill_(0)
        
    if cuda:

        discRealInput = discRealInput.cuda()
        discFakeInput = discFakeInput.cuda()

        discRealLabel = discRealLabel.cuda()
        discFakeLabel = discFakeLabel.cuda()   
        
    print ("Ending training with helper class.")
    print ("Starting training with primary class.")
    
    # training with primary class
    for epoch in range(primaryEpochs):
        for i, data in enumerate(primaryTrainLoader, 0):
            
            #print ('Epoch : {} Primary Class Batch : {}.'.format(epoch+1,i+1))
            
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
            # for every epoch the gradients are reset to 0
            # the discriminator should start to confuse fake primaryInstances
            # with true primaryInstances
            
            G.zero_grad()

            genInputVariable = discFakeInputGen
            genOutputDisc = D(genInputVariable)

            lossGen = lossFunction(genOutputDisc,
                                   discRealLabelVariable)

            lossGen.backward()
            genOptimiser.step()
    
            if (i==0) and epoch==(primaryEpochs-1) :

                folder, primaryClass, _ = primaryFileName.split('_')
                _, helperClass, helperInstances = helperFileName.split('_')
                
                fileName = folder + '_' + str(primaryClass) + '_' + str(helperClass) + '_' + \
                           str(primaryInstances) + '_' + str(helperInstances)
                    
                modelFileName = resultDir+'models/MMD'+'/'+folder+'/'+fileName+'_'+str(epoch)+'.pt'
                plotFileName = resultDir+'plots/MMD'+'/'+folder+'/'+fileName+'_'+str(epoch)+'.png'

                # save the model parameters in a file
                torch.save(G.state_dict(), modelFileName)

                # generate samples from trained generator
                genImage = G(fixedNoiseVariable)
                genImage = genImage.data
                genImage = genImage.cpu()
                genImage = torchvision.utils.make_grid(genImage, nrow=5)
                
                genImage = genImage/2 + 0.5
                genImage = genImage.permute(1,2,0)
                genImage = genImage.numpy()

                # plot the figure of generated samples and save
                fig = plt.figure()
                                
                plt.imshow(genImage, cmap='gray')
                plt.axis('off')

                txt = 'Epoch: '+ str(epoch)
                fig.text(.45,.05,txt)
                if showImage==1:
                    plt.show()
                '''
                IPython.display.clear_output(wait=True)
                IPython.display.display(plt.gcf())
                '''
                plt.savefig(plotFileName, bbox_inches='tight')
                plt.close('all')

    print ("Done trining with primary class primaryInstances.")

def trainSamples( primaryDatasets, primaryClasses, primaryInstances, helperInstances):
    
    
    for dataSet in primaryDataSets:
        for cls in primaryClasses:
            for instance in primaryInstances:
                for helperInstance in helperInstances:
                    
                    # if the number of primary instances are larger than the number of helper
                    # instances, no need to calculate MMD
                    if instance > helperInstance:
                        continue
                    
                    # get a fixed helper class for a particular dataset and primary class
                    helperClass = getHelperClass(dataSet,cls)
                    
                    if helperClass==-1:
                        continue
                    print ('Primary Class: {} Helper Class: {} Primary Instances: {} Helper Instance {}'.
                          format(cls,helperClass,instance,helperInstance))

                    primaryFileName = dataSet+'_'+str(cls)+'_'+str(instance)
                    helperFileName = dataSet+'_'+str(helperClass)+'_'+str(helperInstance)
                    dataFolder = rootDir + str(dataSet)
                    
                    x = loadDataset(dataSet, cls, instance)
                    y = loadDataset(dataSet, helperClass, helperInstance)
                    primaryTrainLoader=torch.utils.data.DataLoader(x,
                                                                   batch_size=batchSize, 
                                                                   shuffle=True,
                                                                   num_workers=4,
                                                                   drop_last = True)
                    helperTrainLoader=torch.utils.data.DataLoader(y,
                                                                  batch_size=batchSize, 
                                                                  shuffle=True,
                                                                  num_workers=4,
                                                                  drop_last=True)
                    numOutputChannels = getChannels(dataSet)
                    epochs = getEpochs(dataSet, instance)

                    print epochs
                    train(primaryFileName, helperFileName, 
                          primaryTrainLoader, helperTrainLoader, 
                          instance, numOutputChannels, 
                          epochs=epochs)               
                    


if __name__=='__main__':
    

    primaryDataSets = ['CIFAR']
    primaryClasses = [0,1,2,3,4,5,6,7,8,9]
    primaryInstances = [100,500,1000,5000]
    helperInstances = [1000, 5000]
    trainSamples(primaryDataSets, primaryClasses, primaryInstances, helperInstances)                    