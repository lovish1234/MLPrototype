import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, numInputChannels, numGenFilter, numOutputChannels):
        super(Generator, self).__init__()
        # Deconvolution layer with <input_channel, output_channel,
        #                            kernel, stride, padding >
        # why is the bias False 
        # state size: 10x1x1
        self.layer1 = nn.ConvTranspose2d(numInputChannels,
                                         numGenFilter*8,
                                         4,
                                         1,
                                         0,
                                         bias = False)
        # state size: 512x4x4
        self.layer2 = nn.BatchNorm2d(numGenFilter*8)
        self.layer3 = nn.ConvTranspose2d(numGenFilter*8,
                                         numGenFilter*4,
                                         4,
                                         2,
                                         1,
                                         bias = False)
        # state size: 256x8x8
        self.layer4 = nn.BatchNorm2d(numGenFilter*4)
        self.layer5 = nn.ConvTranspose2d(numGenFilter*4,
                                         numGenFilter*2,
                                         4,
                                         2,
                                         1,
                                         bias = False)
        # state size: 128x16x16

        self.layer6 = nn.BatchNorm2d(numGenFilter*2)
        self.layer7 = nn.ConvTranspose2d(numGenFilter*2,
                                         numGenFilter,
                                         4,
                                         2,
                                         1,
                                         bias = False)
        # state size: 64x32x32
        self.layer8 = nn.BatchNorm2d(numGenFilter)
        self.layer9 = nn.ConvTranspose2d(numGenFilter,
                                         numOutputChannels,
                                         4,
                                         2,
                                         1,
                                         bias = False)
        # state size 1x64x64

    def forward(self, x):
        x = F.relu(self.layer2(self.layer1(x)),
                   inplace=True)
        x = F.relu(self.layer4(self.layer3(x)),
                   inplace=True)
        x = F.relu(self.layer6(self.layer5(x)),
                   inplace=True)
        x = F.relu(self.layer8(self.layer7(x)),
                   inplace=True)
        x = F.tanh(self.layer9(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, numOutputChannels, numDiscFilter):
        super(Discriminator, self).__init__()

        # state size 1x64x64
        self.layer1 = nn.Conv2d(numOutputChannels,
                                numDiscFilter,
                                4,
                                2,
                                1,
                                bias=False)

        # state size 64x32x32
        self.layer2 = nn.Conv2d(numDiscFilter,
                                numDiscFilter*2,
                                4,
                                2,
                                1,bias=False)
        self.layer3 = nn.BatchNorm2d(numDiscFilter*2)

        # state size 128x16x16
        self.layer4 = nn.Conv2d(numDiscFilter*2,
                                numDiscFilter*4,
                                4,
                                2,
                                1,
                                bias=False)
        self.layer5 = nn.BatchNorm2d(numDiscFilter*4)

        # state size 256x8x8
        self.layer6 = nn.Conv2d(numDiscFilter*4,
                                numDiscFilter*8,
                                4,
                                2,
                                1,
                                bias=False)
        self.layer7 = nn.BatchNorm2d(numDiscFilter*8)

        # state size 512x4x4
        self.layer8 = nn.Conv2d(numDiscFilter*8,
                                1,
                                4,
                                1,
                                0,
                                bias=False)
        # state size 1x1x1

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x),
                         0.2,
                         inplace=True)
        x = F.leaky_relu(self.layer3(self.layer2(x)),
                         0.2,
                         inplace=True)
        x = F.leaky_relu(self.layer5(self.layer4(x)),
                         0.2,
                         inplace=True)
        x = F.leaky_relu(self.layer7(self.layer6(x)),
                         0.2,
                         inplace=True)
        x = F.sigmoid(self.layer8(x))
        return x.view(-1,1)


