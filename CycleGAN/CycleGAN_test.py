import torch
#from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator
import utils
import argparse
import os
import struct
import random
import numpy as np

from skimage.transform import resize
parser = argparse.ArgumentParser()

parser.add_argument('--cuda', required=False, default=True, type=bool, help='gpu availability')
parser.add_argument('--dataset', required=False, default='MNIST', help='input dataset')
parser.add_argument('--batch_size', type=int, default=5, help='test batch size')

# number of channels in input/output eg. Greyscale,1 or RGB,3 or RGBD,4
parser.add_argument('--input_channels', type=int, default=1, help='number of input channels')
parser.add_argument('--output_channels', type=int, default=1, help='number of output channels')

parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--num_resnet', type=int, default=6, help='number of resnet blocks in generator')
#parser.add_argument('--input_size', type=int, default=256, help='input size')

# arguments for changing the input 
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=64, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=False, help='random fliplr True of False')

parser.add_argument('--num_epochs', type=int, default=5, help='number of train epochs')

parser.add_argument('--train_A', type=int, default=5, help='training data for domain A')
parser.add_argument('--train_B', type=int, default=5, help='training data for domain B')

parser.add_argument('--domain_A', type=int, default=1, help='domain data for domain A')
parser.add_argument('--domain_B', type=int, default=7, help='domain data for domain B')



params = parser.parse_args()
print(params)


def gpuAvailable(entity, flag):
    '''
    Change torch.Tensor to torch.CudaTensor
    if a GPU is available
    '''
    if flag:
        entity = entity.cuda()
    return entity


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

def loadDataset(dataSet, classes, numberOfInstances, sizeOfImage, flip=False):
    '''
    Load a particular dataset for testing GAN
    '''
    if dataSet=='MNIST':
        folder = '../MLPrototype/data/MNIST/raw/'
    elif dataSet=='FashionMNIST':
        folder = '../MLPrototype/data/FashionMNIST/raw/'
    elif dataSet=='notMNIST':
        folder = '../MLPrototype/data/notMNIST/raw/'

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

    # shuffle the data instances
    zippedArray = zip(images, labels)
    np.random.shuffle(zippedArray)
    images, labels = zip(*zippedArray)
    images = np.asarray(images)
    labels = np.asarray(labels)

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

    # normalise the image

    # resize images, code may be improved
    imagesByClassResize = np.zeros((imagesByClass.shape[0],
                                    imagesByClass.shape[1],
                                    sizeOfImage,
                                    sizeOfImage))

    # resizing the image to desired size
    for i in range(imagesByClass.shape[0]):
        for j in range(imagesByClass.shape[1]):
            imagesByClassResize[i,j,:,:] = resize(imagesByClass[i,j,:,:], (sizeOfImage,sizeOfImage))

    # flip the image instance around y-axis 
    if flip == True:
        flips = [(slice(None,None,None), slice(None,None,None),
                  slice(None,None,random.choice([-1,None]))) for _ in xrange(numberOfInstances)]
        imagesByClassResize = np.array([image[flip] for image,flip in zip(imagesByClassResize, flips)])

    # convert numpy array to tensor
    imagesByClassResize = torch.from_numpy(imagesByClassResize)
    labelsByClass = torch.from_numpy(labelsByClass)

    # load in a torch Tensor
    data = torch.utils.data.TensorDataset(imagesByClassResize,
                                          labelsByClass)
    trainLoader = torch.utils.data.DataLoader(data,
                                              batch_size = params.batch_size,
                                              shuffle = True,
                                              num_workers = 2)

    return trainLoader, data


# Directories for loading data and saving results
data_dir = '../Data/' + params.dataset + '/'
save_dir = params.dataset+'_'+str(params.domain_A)+'_'+str(params.domain_B) + '_' + str(params.train_A) + '_' + str(params.train_B) + '_' + str(params.num_epochs) + '_results/'
model_dir = params.dataset +'_'+str(params.domain_A)+'_'+str(params.domain_B)+ '_' + str(params.train_A) + '_' + str(params.train_B) + '_' + str(params.num_epochs) + '_model/'


if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Data pre-processing
'''
transform = transforms.Compose([transforms.Scale(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
'''

# Test data
'''
test_data_A = DatasetFromFolder(data_dir, subfolder='testA', transform=transform)
test_data_loader_A = torch.utils.data.DataLoader(dataset=test_data_A,
                                                 batch_size=params.batch_size,
                                                 shuffle=False)
test_data_B = DatasetFromFolder(data_dir, subfolder='testB', transform=transform)
test_data_loader_B = torch.utils.data.DataLoader(dataset=test_data_B,
                                                 batch_size=params.batch_size,
                                                 shuffle=False)
'''

test_data_loader_A,test_data_A = loadDataset(params.dataset,params.domain_A,
                                             100,params.crop_size,params.fliplr)

test_data_loader_B,test_data_B = loadDataset(params.dataset,params.domain_B,
                                             100,params.crop_size,params.fliplr)




# Load model
G_A = Generator(params.input_channels, params.ngf, params.output_channels, params.num_resnet)
G_A.double()
G_B = Generator(params.input_channels, params.ngf, params.output_channels, params.num_resnet)
G_B.double()

G_A = gpuAvailable(G_A, params.cuda)
G_B = gpuAvailable(G_B, params.cuda)
G_A.load_state_dict(torch.load(model_dir + 'generator_A_param.pkl'))
G_B.load_state_dict(torch.load(model_dir + 'generator_B_param.pkl'))


# Test
for i, real_A in enumerate(test_data_loader_A):

    print (len(real_A))
    # input image data
    real_A=real_A[0]
    print (len(real_A))
    print (real_A.shape)

    real_A = Variable(gpuAvailable(real_A, params.cuda))

    # A -> B -> A
    fake_B = G_A(real_A)
    recon_A = G_B(fake_B)

    # Show result for test data
    utils.plot_test_result(real_A, fake_B, recon_A, i, save=True, save_dir=save_dir + 'AtoB/')

    print('%d images are generated.' % (i + 1))

for i, real_B in enumerate(test_data_loader_B):

    # input image data
    real_B = real_B[0]
    real_B = Variable(gpuAvailable(real_B, params.cuda))

    # B -> A -> B
    fake_A = G_B(real_B)
    recon_B = G_A(fake_A)

    # Show result for test data
    utils.plot_test_result(real_B, fake_A, recon_B, i, save=True, save_dir=save_dir + 'BtoA/')

    print('%d images are generated.' % (i + 1))
