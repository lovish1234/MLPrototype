# borrows heavily from yunjay/domainA-domainB-transfer

import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np

from torch.backends import cudnn
from torchvision import transforms

# load the cycleGAN model
from torch.autograd import Variable
from torch import optim
from cycle_model import G12, G21
from cycle_model import D1, D2

# add the loadDataset function [required as an interface to custom dataloaders]
import sys
sys.path.append('../')
from Datasets.loadDataset import loadDataset, getChannels, checkAndCreateFolder
reload(sys.modules[loadDataset.__module__])

# visualization
import matplotlib
import matplotlib.pyplot as plt

import argparse

# saving images in numpy format
from numpy import save as np_save
from numpy import load as np_load

# saving images
import PIL.Image
from cStringIO import StringIO
from os import fsync

# creating animations
import imageio

def sync(fh):
    """
    This makes sure data is written to disk, so that buffering doesn't influence the timings.
    """
    fh.flush()
    fsync(fh.fileno())
    
def showTrainHist(trainHist):
    
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
    
    return plt

def str2bool(v):
    """
    For argument parsing 
    """
    return v.lower() in ('true')

def to_var( x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data( x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

class Solver(object):
    
    def __init__(self,  variables, domainA_loader, domainB_loader):
        
        self.domainB_loader = domainB_loader
        self.domainA_loader = domainA_loader
        
        self.domainA_channels = getChannels(variables.domainA)
        self.domainB_channels = getChannels(variables.domainB)
        
        self.g12 = None
        self.g21 = None
        self.d1 = None
        self.d2 = None
        
        # optimizer parameters
        self.g_optimizer = None
        self.d_optimizer = None
        self.beta1 = variables.beta1
        self.beta2 = variables.beta2
        self.lr = variables.learningRate
        
        # cycleGAN 
        self.use_reconst_loss = variables.use_reconst_loss
        
        # semi-supervised GAN
        self.use_labels = variables.use_labels
        

        self.numGenFilter = variables.numGenFilter
        self.numDiscFilter = variables.numDiscFilter
        
        self.epochs = variables.epochs
        self.batchSize = variables.batchSize
        self.num_classes = variables.num_classes

        # bookkeeping
        self.log_step = variables.log_step
        self.sample_step = variables.sample_step
        
        self.sample_path = variables.sample_path
        self.model_path = variables.model_path
        self.name = variables.name
        
        self.build_model()
        
    def build_model(self):
        """Builds a generator and a discriminator."""
        
        # initalizing generators
        self.g12 = G12(conv_dim=self.numGenFilter, domainA_channels = self.domainA_channels, domainB_channels = self.domainB_channels)
        self.g21 = G21(conv_dim=self.numGenFilter, domainA_channels = self.domainA_channels, domainB_channels = self.domainB_channels)
        
        # initializing discriminators
        self.d1 = D1(conv_dim=self.numDiscFilter, domainA_channels = self.domainA_channels, use_labels=self.use_labels)
        self.d2 = D2(conv_dim=self.numDiscFilter, domainB_channels = self.domainB_channels, use_labels=self.use_labels)
        

        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = list(self.d1.parameters()) + list(self.d2.parameters())
        
        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])
        
        if torch.cuda.is_available():
            self.g12.cuda()
            self.g21.cuda()
            self.d1.cuda()
            self.d2.cuda()
    
    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batchSize))
        merged = np.zeros([3, row*h, row*w*2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row

            merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
            
        return merged.transpose(1, 2, 0)
    
    
    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        
        domainB_iter = iter(self.domainB_loader)    
        domainA_iter = iter(self.domainA_loader)
        
        
        iter_per_epoch = min(len(domainB_iter), len(domainA_iter))
        
        # fixed domainA and domainB for sampling
        fixed_domainB = to_var(domainB_iter.next()[0])
        fixed_domainA = to_var(domainA_iter.next()[0])
        
        # loss if use_labels = True
        criterion = nn.CrossEntropyLoss()
        
        for step in range(self.epochs+1):
            # reset data_iter for each epoch
            if (step+1) % iter_per_epoch == 0:
                domainA_iter = iter(self.domainA_loader)
                domainB_iter = iter(self.domainB_loader)
            
            # load domainB and domainA dataset
            domainB, s_labels = domainB_iter.next() 
            domainB, s_labels = to_var(domainB), to_var(s_labels).long().squeeze()
            domainA, m_labels = domainA_iter.next()
            domainA, m_labels = to_var(domainA), to_var(m_labels)

            if self.use_labels:
                domainA_fake_labels = to_var(
                    torch.Tensor([self.num_classes]*domainB.size(0)).long())
                domainB_fake_labels = to_var(
                    torch.Tensor([self.num_classes]*domainA.size(0)).long())
            
            #============ train D ============#
            
            # train with real images
            self.reset_grad()
            out = self.d1(domainA)
            if self.use_labels:
                d1_loss = criterion(out, m_labels)
            else:
                d1_loss = torch.mean((out-1)**2)
            
            out = self.d2(domainB)
            if self.use_labels:
                d2_loss = criterion(out, s_labels)
            else:
                d2_loss = torch.mean((out-1)**2)
            
            d_domainA_loss = d1_loss
            d_domainB_loss = d2_loss
            d_real_loss = d1_loss + d2_loss
            d_real_loss.backward()
            self.d_optimizer.step()
            
            # train with fake images
            self.reset_grad()
            fake_domainB = self.g12(domainA)
            out = self.d2(fake_domainB)
            if self.use_labels:
                d2_loss = criterion(out, domainB_fake_labels)
            else:
                d2_loss = torch.mean(out**2)
            
            fake_domainA = self.g21(domainB)
            out = self.d1(fake_domainA)
            if self.use_labels:
                d1_loss = criterion(out, domainA_fake_labels)
            else:
                d1_loss = torch.mean(out**2)
            
            d_fake_loss = d1_loss + d2_loss
            d_fake_loss.backward()
            self.d_optimizer.step()
            
            #============ train G ============#
            
            # train domainA-domainB-domainA cycle
            self.reset_grad()
            fake_domainB = self.g12(domainA)
            out = self.d2(fake_domainB)
            reconst_domainA = self.g21(fake_domainB)
            if self.use_labels:
                g_loss = criterion(out, m_labels) 
            else:
                g_loss = torch.mean((out-1)**2) 

            if self.use_reconst_loss:
                g_loss += torch.mean((domainA - reconst_domainA)**2)

            g_loss.backward()
            self.g_optimizer.step()

            # train domainB-domainA-domainB cycle
            self.reset_grad()
            fake_domainA = self.g21(domainB)
            out = self.d1(fake_domainA)
            reconst_domainB = self.g12(fake_domainA)
            if self.use_labels:
                g_loss = criterion(out, s_labels) 
            else:
                g_loss = torch.mean((out-1)**2) 

            if self.use_reconst_loss:
                g_loss += torch.mean((domainB - reconst_domainB)**2)

            g_loss.backward()
            self.g_optimizer.step()
            
            # print the log info
            if (step+1) % self.log_step == 0:
                print('Step [%d/%d], d_real_loss: %.4f, d_domainA_loss: %.4f, d_domainB_loss: %.4f, '
                      'd_fake_loss: %.4f, g_loss: %.4f' 
                      %(step+1, self.epochs, d_real_loss.data[0], d_domainA_loss.data[0], 
                        d_domainB_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))

            # save the sampled images
            if (step+1) % self.sample_step == 0:
                
                fake_domainB = self.g12(fixed_domainA)
                fake_domainA = self.g21(fixed_domainB)
                
                domainA, fake_domainA = to_data(fixed_domainA), to_data(fake_domainA)
                domainB , fake_domainB = to_data(fixed_domainB), to_data(fake_domainB)
                                
                merged = self.merge_images(domainA, fake_domainB)
                path = os.path.join(self.sample_path, self.name)
                checkAndCreateFolder(path)
                path = os.path.join(self.sample_path, self.name, 'sample-%d-m-s.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
                
                merged = self.merge_images(domainB, fake_domainA)
                path = os.path.join(self.sample_path, self.name, 'sample-%d-s-m.png' %(step+1))
               
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
            
            if (step+1) % self.log_step == 0:
                # save the model parameters for each epoch
                g12_path = os.path.join(self.model_path, self.name)
                checkAndCreateFolder(g12_path)
                g12_path = os.path.join(self.model_path, self.name,  'g12-%d.pkl' %(step+1))
                g21_path = os.path.join(self.model_path, self.name, 'g21-%d.pkl' %(step+1))
                d1_path = os.path.join(self.model_path, self.name, 'd1-%d.pkl' %(step+1))
                d2_path = os.path.join(self.model_path, self.name, 'd2-%d.pkl' %(step+1))
                torch.save(self.g12.state_dict(), g12_path)
                torch.save(self.g21.state_dict(), g21_path)
                torch.save(self.d1.state_dict(), d1_path)
                torch.save(self.d2.state_dict(), d2_path)
         
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--numGenFilter', type=int, default=64)
    parser.add_argument('--numDiscFilter', type=int, default=64)
    
    # cycleGAN
    parser.add_argument('--use_reconst_loss', required=True, type=str2bool)
    
    # semi supervised cycleGAN
    parser.add_argument('--use_labels', required=True, type=str2bool)
    
    # for optimizer
    parser.add_argument('--learningRate', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # others
    parser.add_argument('--epochs', type=int, default=40000)
    parser.add_argument('--batchSize', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=10)


    parser.add_argument('--model_path', type=str, default='./CycleGAN/models')
    parser.add_argument('--sample_path', type=str, default='./CycleGAN/samples')
    
    parser.add_argument('--log_step', type=int , default=1000)
    parser.add_argument('--sample_step', type=int , default=1000)
    
    # datasets
    parser.add_argument('--domainA', type=str, default='MNIST')
    parser.add_argument('--domainB', type=str, default='SVHN')
    
    # number of instances in each class of domain A and B
    parser.add_argument('--numDomainA', type=int, default=6000)
    parser.add_argument('--numDomainB', type=int, default=6000)
    
    parser.add_argument('--name', type=str, default='Experiment1')
        
    # dataloader
    
    
    variables = parser.parse_args()
    
    
    # define dataloaders of datasets
    classes = [0,1,2,3,4,5,6,7,8,9]
    
    instances = [variables.numDomainA for i in range(10)]

    
    domainA_dataset = loadDataset(variables.domainA, classes, instances, 'train')
    domainA_dataloader = torch.utils.data.DataLoader(domainA_dataset, 
                                                   batch_size = variables.batchSize,
                                                   shuffle = True,
                                                   num_workers = 2,
                                                   drop_last=True)

    instances = [variables.numDomainB for i in range(10)]


    domainB_dataset = loadDataset(variables.domainB, classes, instances, 'train')
    domainB_dataloader = torch.utils.data.DataLoader(domainB_dataset, 
                                                   batch_size = variables.batchSize,
                                                   shuffle = True,
                                                   num_workers = 2,
                                                   drop_last=True)
    
    # define the Solver
    solver = Solver(variables, domainA_dataloader, domainB_dataloader)
    solver.train()

