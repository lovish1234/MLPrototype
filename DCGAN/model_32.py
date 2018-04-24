import torch
import torch.nn as nn
import torch.nn.functional as F


# for 32x32 images, numGenFilter = 32, numDiscFilter=32
class Generator(nn.Module):
    def __init__(self, numInputChannels, numGenFilter, numOutputChannels):
        super(Generator, self).__init__()
        
        # Deconvolution layer with <input_channel, output_channel,
        #                            kernel, stride, padding >
        
        # state size: 10x1x1
        self.layer1 = nn.ConvTranspose2d(numInputChannels,
                                         numGenFilter*8,
                                         4,
                                         1,
                                         0,
                                         bias = False)
        # state size: 256x4x4
        self.layer2 = nn.BatchNorm2d(numGenFilter*8)
        self.layer3 = nn.ConvTranspose2d(numGenFilter*8,
                                         numGenFilter*4,
                                         4,
                                         2,
                                         1,
                                         bias = False)
        # state size: 128x8x8
        self.layer4 = nn.BatchNorm2d(numGenFilter*4)
        self.layer5 = nn.ConvTranspose2d(numGenFilter*4,
                                         numGenFilter*2,
                                         4,
                                         2,
                                         1,
                                         bias = False)
        # state size: 64x16x16

        self.layer6 = nn.BatchNorm2d(numGenFilter*2)
        self.layer7 = nn.ConvTranspose2d(numGenFilter*2,
                                         numOutputChannels,
                                         4,
                                         2,
                                         1,
                                         bias = False)
        # state size: 3x32x32

    def forward(self, x):
        x = F.relu(self.layer2(self.layer1(x)),
                   inplace=True)
        x = F.relu(self.layer4(self.layer3(x)),
                   inplace=True)
        x = F.relu(self.layer6(self.layer5(x)),
                   inplace=True)
        x = F.tanh(self.layer7(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, numOutputChannels, numDiscFilter):
        super(Discriminator, self).__init__()

        # state size 3x32x32
        self.layer1 = nn.Conv2d(numOutputChannels,
                                numDiscFilter*2,
                                4,
                                2,
                                1,
                                bias=False)

        # state size 64x16x16
        self.layer2 = nn.Conv2d(numDiscFilter*2,
                                numDiscFilter*4,
                                4,
                                2,
                                1,bias=False)
        self.layer3 = nn.BatchNorm2d(numDiscFilter*4)

        # state size 128x8x8
        self.layer4 = nn.Conv2d(numDiscFilter*4,
                                numDiscFilter*8,
                                4,
                                2,
                                1,
                                bias=False)
        self.layer5 = nn.BatchNorm2d(numDiscFilter*8)

        # state size 256x4x4
        self.layer6 = nn.Conv2d(numDiscFilter*8,
                                1,
                                4,
                                1,
                                0,
                                bias=False)

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
        x = F.sigmoid(self.layer6(x))
        return x.view(-1,1)


