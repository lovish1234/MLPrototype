import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        
# for 32x32 images, numGenFilter = 32, numDiscFilter=32
class Generator(nn.Module):
    def __init__(self, numInputChannels, numLabels, numGenFilter, numOutputChannels):
        super(Generator, self).__init__()
        
        # Deconvolution layer with <input_channel, output_channel,
        #                            kernel, stride, padding >
        
        # state size: 10x1x1
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
                                         numOutputChannels,
                                         4,
                                         2,
                                         1,
                                         bias = False)
        # state size: 3x32x32
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
    def forward(self, x, c):
        
        x = F.relu(self.layer1_BN(self.layer1(x)),
                   inplace=True)
        
        c = F.relu(self.layer1Label_BN(self.layer1Label(c)),
                   inplace=True)
        
        x = torch.cat([x,c], 1)
        
        x = F.relu(self.layer2_BN(self.layer2(x)),
                   inplace=True)
        x = F.relu(self.layer3_BN(self.layer3(x)),
                   inplace=True)
        x = F.tanh(self.layer4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, numLabels):
        super(Discriminator, self).__init__()

        # state size 3x32x32
        
        self.layer1 = nn.Linear(84, 32)
        self.layer1Label = nn.Linear(numLabels,32)
        
        self.layer2 = nn.Linear(64, 32)
        self.layer2_BN = nn.BatchNorm1d(32)
        
        self.layer3 = nn.Linear(32,16)
        self.layer3_BN = nn.BatchNorm1d(16)
        
        self.layer4 = nn.Linear(16,1)
        


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
    def forward(self, x, c):
        
        x = F.leaky_relu(self.layer1(x),
                         0.2,
                         inplace=True)
        c = F.leaky_relu(self.layer1Label(c),
                         0.2,
                         inplace=True)
        
        x = torch.cat([x,c],1)
        
        x = F.leaky_relu(self.layer2_BN(self.layer2(x)),
                         0.2,
                         inplace=True)
        x = F.leaky_relu(self.layer3_BN(self.layer3(x)),
                         0.2,
                         inplace=True)
        
        x = F.sigmoid(self.layer4(x))
        return x.view(-1,1)

