import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, numInputChannels, numLabels, numGenFilter, numOutputChannels):
        super(Generator, self).__init__()
        # Deconvolution layer with <input_channel, output_channel,
        #                            kernel, stride, padding >
        # why is the bias False 
        # state size: 10x1x1
        
        # state size 1x64x64
        super(Generator, self).__init__()
        

        self.layer1 = nn.ConvTranspose2d(numInputChannels,
                                         numGenFilter*4,
                                         4,
                                         1,
                                         0,
                                         bias = False)
        # state size: 256x4x4
        self.layer1_BN = nn.BatchNorm2d(numGenFilter*4)
        
        self.layer1Label = nn.ConvTranspose2d(numLabels,
                                         numGenFilter*4,
                                         4,
                                         1,
                                         0,
                                         bias = False)
        self.layer1Label_BN = nn.BatchNorm2d(numGenFilter*4)
        
        
        self.layer2 = nn.ConvTranspose2d(numGenFilter*8,
                                         numGenFilter*4,
                                         4,
                                         2,
                                         1,
                                         bias = False)
        self.layer2_BN = nn.BatchNorm2d(numGenFilter*4)
        
        # state size: 128x8x8

        self.layer3 = nn.ConvTranspose2d(numGenFilter*4,
                                         numGenFilter*2,
                                         4,
                                         2,
                                         1,
                                         bias = False)
        self.layer3_BN = nn.BatchNorm2d(numGenFilter*2)
        
        # state size: 64x16x16

        self.layer4 = nn.ConvTranspose2d(numGenFilter*2,
                                         numGenFilter,
                                         4,
                                         2,
                                         1,
                                         bias = False)
        self.layer4_BN = nn.BatchNorm2d(numGenFilter*2)
        # state size: 3x32x32
        
        self.layer5 = nn.ConvTranspose2d(numGenFilter,
                                         numOutputChannels,
                                         4,
                                         2,
                                         1,
                                         bias = False)   
        
    def forward(self, x):
        
        x = F.relu(self.layer1_BN(self.layer1(x)),
                   inplace=True)
        c = F.relu(self.layer1Label_BN(self.layer1(c)),
                   inplace=True)
        
        x = torch.cat([x,c], 1)
        
        x = F.relu(self.layer2_BN(self.layer2(x)),
                   inplace=True)
        x = F.relu(self.layer3_BN(self.layer3(x)),
                   inplace=True)
        x = F.relu(self.layer4_BN(self.layer4(x)),
                   inplace=True)
        x = F.tanh(self.layer5(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, numOutputChannels, numLabels, numDiscFilter):
        super(Discriminator, self).__init__()

        # state size 3x32x32
        self.layer1 = nn.Conv2d(numOutputChannels,
                                numDiscFilter/2,
                                4,
                                2,
                                1,
                                bias=False)
        
        self.layer1Label = nn.Conv2d(numLabels,
                                numDiscFilter/2,
                                4,
                                2,
                                1,
                                bias=False)

        # state size 64x16x16
        self.layer2 = nn.Conv2d(numDiscFilter,
                                numDiscFilter*2,
                                4,
                                2,
                                1,bias=False)
        self.layer2_BN = nn.BatchNorm2d(numDiscFilter*2)

        # state size 128x8x8
        self.layer3 = nn.Conv2d(numDiscFilter*2,
                                numDiscFilter*4,
                                4,
                                2,
                                1,
                                bias=False)
        self.layer3_BN = nn.BatchNorm2d(numDiscFilter*4)

        # state size 256x4x4
        self.layer4 = nn.Conv2d(numDiscFilter*4,
                                numDiscFilter*8,
                                4,
                                1,
                                0,
                                bias=False)
        self.layer4_BN = nn.BatchNorm2d(numDiscFilter*8)
        
        self.layer5 = nn.Conv2d(numDiscFilter*8,
                                1,
                                4,
                                1,
                                0,
                                bias=False)


    def forward(self, x):

        x = F.leaky_relu(self.layer1(x),
                         0.2,
                         inplace=True)
        c = F.leaky_relu(self.layerLabel(c),
                         0.2,
                         inplace=True)
        
        x = torch.cat([x,c], 1)
        
        x = F.leaky_relu(self.layer2_BN(self.layer2(x)),
                         0.2,
                         inplace=True)
        x = F.leaky_relu(self.layer3_BN(self.layer3(x)),
                         0.2,
                         inplace=True)
        x = F.leaky_relu(self.layer4_BN(self.layer4(x)),
                         0.2,
                         inplace=True)
        
        x = F.sigmoid(self.layer5(x))
        return x.view(-1,1)


