#TBD
# PEP8 Linting
# Argparse 
#/usr/bin/env python
from  __future__ import print_function
import numpy as np
import torch

# neural network class
import torch.nn as nn

# activation functions  
import torch.nn.functional as F

# different optimisation techniques 
import torch.optim as optim

# support for automatically getting the gradients 
from torch.autograd import Variable

import  matplotlib.pyplot as plt


# Debug 
debug = 0


# real data paramters
mean = 4
variance = 1.25

# generator parameters 
genInputSize = 1
genHiddenSize = 50
genOutputSize = 1
genIter = 1

# discriminator parameters   
discInputSize = 100
discHiddenSize = 50
discOutputSize = 1
discIter = 1

# discriminator output weather the generator is producing real/fake data
miniBatchSize = discInputSize

# optimiser parameters ( adam optimiser )
learningRate = 2e-4
optimBetas = (0.9,0.999)
epochs = 30000 

# other parameters
print_stat = 200 

# initial noise vector defined as a uniform distribution here
def noise_samples(M,N):
    return torch.rand(M,N)


# real samples need to be defined as gaussian
# need to convert it in a Tensor before passing on to a Pytorch Variable 
def real_samples(mean, variance, N):
    return torch.Tensor(np.random.normal(mean, variance, (1,N)))

class Generator(nn.Module):
    def __init__(self):
        # do not understand super 
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(genInputSize, genHiddenSize)
        self.layer2 = nn.Linear(genHiddenSize, genHiddenSize)
        self.layer3 = nn.Linear(genHiddenSize, genOutputSize)

    def forward(self, inputMatrix):
        inputMatrix = inputMatrix.view(-1,genInputSize)
        x = F.elu(self.layer1(inputMatrix))
        x = F.sigmoid(self.layer2(x))
        x = self.layer3(x)
        return x

class Discriminator(nn.Module):
    # contains more non linearity to make it stronger than the generator 
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(discInputSize, discHiddenSize)
        self.layer2 = nn.Linear(discHiddenSize, discHiddenSize)
        self.layer3 = nn.Linear(discHiddenSize, discOutputSize)


    def forward(self, inputMatrix):
        inputMatrix = inputMatrix.view(-1,discInputSize)
        x = F.elu(self.layer1(inputMatrix))
        x = F.elu(self.layer2(x))
        x = F.sigmoid(self.layer3(x))
        return x


def convertToList(tensor):
    return tensor.data.storage().tolist()

def findMetrics(tensor):
    return [np.mean(convertToList(tensor)), np.std(convertToList(tensor))]

def train():
    # create an instance of both discriminator and generator
    D = Discriminator()
    G = Generator()

    # define a optimisation method to be used 
    discOptimiser = optim.Adam(D.parameters(), lr=learningRate, betas = optimBetas)
    genOptimiser = optim.Adam(G.parameters(), lr=learningRate, betas = optimBetas)

    # using the binary cross-entropy loss function 
    lossFunction = nn.BCELoss()
    discRealLossList = []
    discFakeLossList = []
    genLossList = []

    for epoch in range(epochs):

        for i in range (discIter):
            # for each step, set the gradient to zero
            D.zero_grad()

            # train discriminator on real data
            discRealInput = Variable(real_samples(mean,variance,discInputSize))
            if debug:
                print (discRealInput.size())
            discRealOutput = D(discRealInput)
            if debug:
                print (discRealOutput.size())

            # loss function 
            target = Variable(torch.ones(discOutputSize))
            discRealLoss = lossFunction(discRealOutput, target)
            if debug:
                print(discRealLoss.size())
                print (discRealLoss)
            discRealLoss.backward()

            # train discriminacor on fake data
            discFakeInput = Variable(noise_samples(discInputSize,genInputSize))
            if debug:
                print (discFakeInput.size())


            discFakeInput = G(discFakeInput).detach() #??          
            if debug:
                print (discFakeInput.size())

            discFakeOutput = D(discFakeInput)
            if debug:
                print (discFakeOutput.size())

            # loss function
            target = Variable(torch.zeros(discOutputSize))
            discFakeLoss = lossFunction(discFakeOutput, target)
            discFakeLoss.backward()


            # backpropagation for discriminator 
            # update the weights 
            discOptimiser.step()

        for i in range(genIter):
            # for each step, set the gradient to zero
            G.zero_grad()

            # train generator
            # generator takes input of the form 1x1 as input 
            # miniBatchSize = 1 and dimension = 1
            genInput = Variable(noise_samples(miniBatchSize, genInputSize))
            if debug:
                print (genInput.size())
            genOutput = G(genInput)
            if debug:
                print (genOutput.size())
            # generator gives the image to generator for prediction
            discOutput = D(genOutput)
            if debug:
                print (discOutput.size())

            # to fool the discriminator, all the samples are genuine
            target = Variable(torch.ones(genOutputSize))

            #loss function and backpropagation
            genLoss = lossFunction(discOutput, target)
            if debug:
                print (genLoss.size())


            genLoss.backward()
            # update the weights 
            genOptimiser.step()

        listDiscRealLoss = convertToList(discRealLoss)
        listDiscFakeLoss = convertToList(discFakeLoss)
        listGenLoss = convertToList(genLoss)

        discRealLossList.append(listDiscFakeLoss[0])
        discFakeLossList.append(listDiscRealLoss[0])
        genLossList.append(listGenLoss[0])

        if epoch % print_stat == 0:
            print  ("%s D_Real:%s D_Fake:%s G:%s StatReal:%s StatFake:%s" %
                    (epoch,
                    listDiscRealLoss,
                    listDiscFakeLoss,
                    listGenLoss,
                    findMetrics(discRealInput),
                    findMetrics(genOutput)
                    ))
    plt.plot(discRealLossList, color='green', alpha=0.5, label='Discriminator Real')
    plt.plot(discFakeLossList, color='blue', alpha=0.5, label='Discriminator Fake')
    plt.plot(genLossList, color='red', alpha=0.5, label='Generator')
    plt.legend(loc = 'upper right')
    name = 'uniformtoGaussianLoss'+str(epochs)+'.png' 
    plt.savefig(name)
    plt.show()
    plot_distribution(discRealInput, genInput, genOutput)


def plot_distribution(discRealInput, genInput, genOutput):

    # plot the distribution for real samples, must be Gaussian
    count, bins, ignored = plt.hist(convertToList(discRealInput), 30,
            normed=True, color = 'green', alpha=0.5, label='Real Distribution')

    # plot the distribution for noise, must be uniform
    count, bins, ignored = plt.hist(convertToList(genInput), 30, normed=True,
            color = 'blue', alpha=0.5, label='Noise')

    # the generated distribution must be similar to real distribution
    count, bins, ignored = plt.hist(convertToList(genOutput), 30, normed=True,
            color = 'red', alpha=0.5, label='Generated Distribution')

    plt.legend(loc = 'upper right')

    name = 'uniformToGaussian'+str(epochs)+'.png'
    plt.savefig(name)

    plt.show()
if __name__=="__main__":
    #plot_distribution()
    train()

