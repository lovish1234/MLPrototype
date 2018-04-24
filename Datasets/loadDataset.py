import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import torchvision.utils as utils

import math
import numpy as np

from PIL import Image
import PIL.ImageOps 

import sys
import pickle

import os
import os.path
import numpy as np

import errno
import codecs

import pandas

# plotting
import matplotlib.pyplot as plt

# classifiers
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

# random shuffling
import random


# import configuration
sys.path.append('../DCGAN/')
import config
reload(config)
from config import *

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


# check and create a folder if it does not exist 
def checkAndCreateFolder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as err:
            if err.errno==errno.EEXIST:
                raise
                
# show samples of a particular images
def getImageSamples(dataSet, array, numOfChannels=1, sampleSize=25, nRows=5):
    '''
    array : 
    sampleSize :
    nRows :
    
    returns a maximum of 25 dataset samples in a grid.
    '''
    numOfImages = array.shape[0]
    imageSize = int(math.sqrt(array.shape[1]/numOfChannels))
    
    sampleSize = min(sampleSize, numOfImages)
    array = array[:sampleSize]
    
    sampleImage = torch.from_numpy(array.reshape(array.shape[0], numOfChannels, imageSize, imageSize))
    sampleImage = utils.make_grid(sampleImage, nrow = nRows)
    sampleImage = (sampleImage/2)+0.5
    #sampleImage = unnormalizeImage(sampleImage, dataSet)
    
    sampleImage = sampleImage.permute(1,2,0)
    sampleImage = sampleImage.numpy()

    return sampleImage
   


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# for birds dataset loading
def get_imgs(img_path, bboxCrop=False, bbox=None):
    '''
    
    '''

    img = Image.open(img_path).convert('RGB')
    
    if bboxCrop and bbox is not None:
        lower_x, lower_y = bbox[0], bbox[1]
        upper_x, upper_y = bbox[2]+bbox[0], bbox[3]+bbox[1]
        croppedImg = img.crop([lower_x, lower_y, upper_x, upper_y])
        
        #simg = cimg.resize((64,64), Image.ANTIALIAS)
        
        return croppedImg
    else:
        return img
    

# function corresponding to extracting *MNIST type datasets
def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def toVariable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def getAttributes(primaryInstances, 
                  helperInstances, 
                  fewShotInstances, 
                  primaryClasses, 
                  helperClasses, 
                  fewShotClasses):
    '''
    primaryInstances - Total number of instances per class in primary domain
    helperInstances -  Total number of instances per class in helper domain
    fewShotInstances - Number of instances pertaining few shot classes
    
    primaryClasses - Number of classes in primary domain
    helperClasses - Number of classes in helper Domain
    fewShotClasses - List of classes where limited data is available
    
    return classes_ and instances_ list for primary and helper domain
    '''
    
    primaryClassList=[]
    primaryInstanceList=[]
    
    helperClassList=[]
    helperInstanceList=[]
    
    for i in range(helperClasses):
        helperClassList.append(i)
        helperInstanceList.append(helperInstances)
        
    for i in range(primaryClasses):
        primaryClassList.append(i)
        if i in fewShotClasses:
            primaryInstanceList.append(fewShotInstances)
        else:
            primaryInstanceList.append(primaryInstances)
    
        
    return primaryClassList, primaryInstanceList, helperClassList, helperInstanceList  


# functions to get dataset attributes

def getImageSize(dataSet):
    '''
    '''
    if dataSet=='MNIST' or \
       dataSet=='notMNIST' or \
       dataSet=='FashionMNIST' or \
       dataSet=='negativeMNIST' or \
       dataSet=='rotatedMNIST':
        return 28
    elif dataSet=='SVHN' or \
         dataSet=='CIFAR' or \
         dataSet=='LSUN':
        return 32
    elif dataSet=='NABirds' or \
         dataSet=='CUBS':
        return 32
    elif dataSet=='USPS':
        return 16
       

# functions correseponding to particular datasets
def getChannels(dataSet):
    if dataSet=='MNIST' or \
       dataSet=='notMNIST' or \
       dataSet=='FashionMNIST' or \
       dataSet=='negativeMNIST' or \
       dataSet=='rotatedMNIST' or \
       dataSet=='SVHN-BW':
        return 1
    elif dataSet=='SVHN' or \
         dataSet=='CIFAR' or \
         dataSet=='LSUN' or \
         dataSet=='CUBS' or \
         dataSet=='NABirds':
        return 3

def normalizeImage(tensor, dataSet):
    
    mean, std = getDataSetMoments(dataSet)
    
    for t,m,s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def unnormalizeImage(tensor, dataSet):
    
    mean, std = getDataSetMoments(dataSet)
    
    for t,m,s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor    


def getMoments(dataset):
    '''
    Get mean and std for images of these datasets only
    Currently for 'MNIST', 'FashionMNISt', 'CIFAR', 'SVHN' only
    '''
    # take instances irrespective of class
    x = loadDataset(dataset, [0,1,2,3,4,5,6,7,8,9] , [5000,5000,5000,5000,5000,5000,5000,5000,5000,5000] , transform=False )
    dataloader = torch.utils.data.DataLoader(x, batch_size=10000, shuffle=False, num_workers=2)
    numOfChannels = getChannels(dataset)
    train = iter(dataloader).next()[0]

    
    if numOfChannels==3:
        mean = np.mean(train.numpy(), axis=(0, 2, 3))
        std = np.std(train.numpy(), axis=(0, 2, 3))
    else:
        mean = np.mean(train.numpy())
        std = np.std(train.numpy())
    return mean, std

def getDataSetMoments(dataSet):
    
    if dataSet=='CIFAR':
        mean = (0.49140087,  0.48215872,  0.44653106)
        std = (0.24703316,  0.24348569,  0.26158798) 
    elif dataSet=='SVHN':
        mean = (0.43809715,  0.44423839,  0.4731389)
        std = (0.19782759,  0.20084523,  0.19693449)
    elif dataSet=='MNIST':
        mean = (0.13074383,0.13074383,0.13074383)
        std =  (0.30820516,0.30820516,0.30820516)
    elif dataSet=='FashionMNIST':
        mean = (0.28601688,0.28601688,0.28601688)
        std = (0.35292232,0.35292232,0.35292232)
    elif dataSet=='SVHN-BW':
        mean = (0.44200814,0.44200814,0.44200814)
        std = (0.19416408,0.19416408,0.19416408)
    elif dataSet=='NABirds':
        mean = (0.47642726,  0.47663745,  0.43016016)
        std = (0.2090884 ,  0.20856003,  0.24288875)
    elif dataSet=='CUBS':
        mean = (0.4717457 ,  0.47085917,  0.40564072)
        std = (0.2151828 ,  0.21099399,  0.23933935)
    elif dataSet=='USPS':
        # copied MNIST moments, need to calculate
        mean = (0.13074383,0.13074383,0.13074383)
        std =  (0.30820516,0.30820516,0.30820516)        
        
    return mean, std

def getTransforms(dataSet):
    
    mean, std = getDataSetMoments(dataSet)
    
    if dataSet=='CIFAR':
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform
    elif dataSet=='SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform        
    elif dataSet=='MNIST':
        transform = transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform
    elif dataSet=='USPS':
        transform = transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform        
    elif dataSet=='FashionMNIST':
        transform = transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform
    elif dataSet=='notMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])  
        return transform
    elif dataSet=='negativeMNIST':
        transform = transforms.Compose([
            transforms.Lambda(lambda image: PIL.ImageOps.invert(image)),
            transforms.ToTensor(),
            transforms.Normalize((0.86929756,), (0.30816492,))
        ])        
        return transform
    elif dataSet=='rotatedMNIST':
        transform = transforms.Compose([
            transforms.Lambda(lambda image: image.rotate(90)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) 
    elif dataSet=='SVHN-BW':
        transform = transforms.Compose([
            transforms.Scale(32),
            transforms.Lambda(lambda image: image.convert('L')),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
            transforms.Normalize((0.5,), (0.5,))
        ])
        return transform
    elif  dataSet=='NABirds':
        transform = transforms.Compose([
            transforms.Scale((32,32)),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform
    elif dataSet=='CUBS':
        transform = transforms.Compose([
            transforms.Scale((32,32)),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform        
    
# function corresponding to classes
def getHelperClass(dataSet, primaryClass):
    '''
    Set of primary and helper classes
    MNIST
    
    0:9
    1:7
    2:4
    3:8
    4:2
    5:6
    6:5
    7:1
    8:3
    9:0
    
    FashionMNIST
    
    T-Shirt:Shirt
    Trouser:Dress
    Pullover:Coat
    Dress:Trouser
    Coat:Pullover
    Sandal:Coat
    Shirt:T-shirt
    Sneaker:Ankle Boot
    Bag:Ankle Boot
    Ankle Boot: Sneaker
    
    notMNIST
    E:F
    F:E
    
    SVHN
    Same as MNIST
    
    CIFAR
    
    Aeroplane:Bird
    Automobile:Ship
    Bird:Aeroplane
    Cat:Deer
    Deer:Cat
    Dog:Cat
    Frog:Dog
    Horse:Dog
    Ship:Truck
    Truck:Ship
    
    '''
    if dataSet=='MNIST' or dataSet=='negativeMNIST' or dataSet=='rotatedMNIST':
        helpDict = {0:9,1:7,2:4,3:8,4:2,5:6,6:5,7:1,8:3,9:0}
    elif dataSet=='FashionMNIST':
        helpDict = {0:6,1:2,2:4,3:4,4:3,5:7,6:0,7:9,8:9,9:7}
    elif dataSet=='notMNIST':
        helpDict = {0:-1,1:-1,2:-1,3:-1,4:5,5:4,6:-1,7:-1,8:-1,9:-1}
    elif dataSet=='SVHN':
        helpDict = {0:9,1:7,2:4,3:8,4:2,5:6,6:5,7:1,8:3,9:0}
    elif dataSet=='CIFAR':
        helpDict = {0:2,1:8,2:0,3:4,4:3,5:3,6:5,7:5,8:9,9:8}
    else:
        return -1
    return helpDict[primaryClass]

def getEpochs(dataSet, numOfInstances):
    '''
    Inputs:
    
    numOfInstances : Int : Number of instances used from original dataset
    
    Outputs:
    
    epochs : Int : Number of epochs for which Generator has trained

    
    '''
    if dataSet=='MNIST':
        if numOfInstances==10:
            return 500
        elif numOfInstances==100:
            return 2000
        elif numOfInstances==200:
            return 900
        elif numOfInstances==500:
            return 500
        elif numOfInstances==1000:
            return 500
        elif numOfInstances==5000:
            return 100
        elif numOfInstances==10000:
            return 100
        elif numOfInstances==50000:
            return 20
    elif dataSet=='FashionMNIST':
        if numOfInstances==10:
            return 500
        elif numOfInstances==100:
            return 2000
        elif numOfInstances==200:
            return 900
        elif numOfInstances==500:
            return 500
        elif numOfInstances==1000:
            return 500
        elif numOfInstances==5000:
            return 100
        elif numOfInstances==10000:
            return 100
        elif numOfInstances==50000:
            return 20
    elif dataSet=='notMNIST':
        if numOfInstances==10:
            return 500
        elif numOfInstances==100:
            return 500
        elif numOfInstances==200:
            return 400
        elif numOfInstances==500:
            return 250
        elif numOfInstances==1000:
            return 200
        elif numOfInstances==5000:
            return 50
    elif dataSet=='SVHN':
        if numOfInstances==10:
            return 500
        elif numOfInstances==100:
            return 2000
        elif numOfInstances==200:
            return 900
        elif numOfInstances==500:
            return 500
        elif numOfInstances==1000:
            return 500
        elif numOfInstances==5000:
            return 100
        elif numOfInstances==10000:
            return 100
        elif numOfInstances==50000:
            return 20
    elif dataSet=='negativeMNIST':
        if numOfInstances==10:
            return 500
        elif numOfInstances==100:
            return 500
        elif numOfInstances==200:
            return 400
        elif numOfInstances==500:
            return 250
        elif numOfInstances==1000:
            return 200
        elif numOfInstances==5000:
            return 50
    elif dataSet=='rotatedMNIST':
        if numOfInstances==10:
            return 500
        elif numOfInstances==100:
            return 500
        elif numOfInstances==200:
            return 400
        elif numOfInstances==500:
            return 250
        elif numOfInstances==1000:
            return 200
        elif numOfInstances==5000:
            return 50
    elif dataSet=='CIFAR':
        if numOfInstances==10:
            return 500
        elif numOfInstances==100:
            return 2000
        elif numOfInstances==200:
            return 900
        elif numOfInstances==500:
            return 500
        elif numOfInstances==1000:
            return 500
        elif numOfInstances==5000:
            return 100
        elif numOfInstances==10000:
            return 100
        elif numOfInstances==50000:
            return 20
    elif dataSet=='NABirds':
        if numOfInstances==10:
            return 500
        elif numOfInstances==100:
            return 1000
        elif numOfInstances==200:
            return 900
        elif numOfInstances==500:
            return 500
        elif numOfInstances==1000:
            return 200
        elif numOfInstances==5000:
            return 100
        elif numOfInstances==50000:
            return 20
    elif dataSet=='CUBS':
        if numOfInstances<=10:
            return 500
        elif numOfInstances<=100:
            return 500
        elif numOfInstances<=200:
            return 400
        elif numOfInstances<=500:
            return 250
        elif numOfInstances<=1000:
            return 200
        elif numOfInstances<=6000:
            return 50
    elif dataSet=='SVHN-BW':
        if numOfInstances==10:
            return 500
        elif numOfInstances==100:
            return 1000
        elif numOfInstances==200:
            return 900
        elif numOfInstances==500:
            return 500
        elif numOfInstances==1000:
            return 200
        elif numOfInstances==5000:
            return 100   
       
def getClasses(dataSet):
    if dataSet=='MNIST' or \
       dataSet=='negativeMNIST' or \
       dataSet=='rotatedMNIST' or \
       dataSet=='SVHN' or\
       dataSet=='SVHN-BW' or\
       dataSet=='USPS':
        class_names = [0,1,2,3,4,5,6,7,8,9]
    elif dataSet=='FashionMNIST':
        class_names = ['T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    elif dataSet=='notMNIST':
        class_names = ['A','B','C','D','E','F','G','H','I','J']
    elif dataSet=='CIFAR':
        class_names = ['Aeroplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
        
    return class_names

def getClassifier(type_, C=10.0, degree=3, n_estimators=10):
    if type_=='linear':
        return LinearSVC()
    elif type_=='rbf':    
        return SVC(C=C, kernel='rbf')    
    elif type_=='poly':
        return SVC(C=C, kernel='poly', degree=degree) 
    elif type_=='randomForest':
        return RandomForestClassifier(max_depth=2, random_state=0)
    elif type_=='linearBag':
        return BaggingClassifier(SVC(kernel='linear'), max_samples=1.0 / n_estimators, n_estimators=n_estimators)
    elif type_=='rbfBag':
        return BaggingClassifier(SVC(kernel='rbf', C=C), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=2)
    elif type_=='polyBag':
        return BaggingClassifier(SVC(kernel='poly', C=C, degree=degree), 
                                 max_samples=1.0 / n_estimators, n_estimators=n_estimators)
    elif type_=='sgd':
        return SGDClassifier(alpha=0.001, max_iter=100)
    
# loading datasets
def loadDataset(dataSet, classes, instances, mode='train', bboxCrop=False, transform=True):
    
    '''
    dataset : domain
    classes : list of classes to be loaded from the dataset
    instances : list of instnaces of a particular class to be loaded from the dataset
    bboxCrop : In case bounding boxes are available, bboxCrop=True crops the image
    '''
    
    dataFolder = rootDir + '/' + dataSet+'/'
    if transform:
        transform = getTransforms(dataSet)
    else:
        transform = transforms.ToTensor()
        
    if dataSet=='MNIST':
        return MNIST(dataFolder, classes, instances, train=(mode=='train'), transform=transform)
    elif dataSet=='FashionMNIST':
        return FashionMNIST(dataFolder, classes, instances, train=(mode=='train'), transform=transform)
    # Street View House Numbers dataset
    elif dataSet=='SVHN':
        return SVHN(dataFolder, classes, instances, split=mode ,transform=transform)
    elif dataSet=='CIFAR':
        return CIFAR(dataFolder, classes, instances, train=(mode=='train'), transform=transform)
    # inverted MNIST dataset
    elif dataSet=='negativeMNIST':
        dataFolder = rootDir + '/' + 'MNIST/'
        return MNIST(dataFolder, classes, instances, train=(mode=='train'), transform=transform)
        # rotated MNIST dataset
    elif dataSet=='rotatedMNIST':
        dataFolder = rootDir + '/' + 'MNIST/'
        return MNIST(dataFolder, classes, instances, train=(mode=='train'), transform=transform)
    elif dataSet=='SVHN-BW':
        dataFolder = rootDir + '/' + 'SVHN/'
        return SVHN(dataFolder, classes, instances, split=mode ,transform=transform)
    elif dataSet=='NABirds':
        return Birds(dataFolder, instances, split=mode ,transform=transform, bboxCrop=bboxCrop) 
    elif dataSet=='CUBS':
        return Birds(dataFolder, instances, split=mode ,transform=transform, bboxCrop=bboxCrop)     
    elif dataSet=='notMNIST':
        return notMNIST(dataFolder, classes, instances, transform=transform)
    elif dataSet=='USPS':
        return USPS(dataFolder, classes, instances, train=(mode=='train'),transform=transform)
        
    else:
        print ('Dataset {} could not be loaded.'.format(dataSet))

    
class SVHN(datasets.SVHN):

    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, classes_, instances_, split='train',
                 transform=None, target_transform=None, 
                 download=False):
        '''
        class_: list of fewShotClasses, the classes for which limited data is available
        '''
        self.classes_=classes_
        self.instances_ = instances_
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set
        

        
        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        
        self.data = np.transpose(self.data, (3, 2, 0, 1))
                

        # select the data from a particular class only
        #self.data = self.data[np.where(self.labels==self.class_)]
        #self.labels = self.labels[np.where(self.labels==self.class_)]

        #self.data = np.delete(self.data, np.where(self.labels==c), axis=0)               
        #self.labels = np.delete(self.labels, np.where(self.labels==c), axis=0)
        
        dataArray = np.array([])
        labelArray = np.array([])
        
        # randomize and get required number of instances
        for i in range(len(classes_)):
           
            tempData = self.data[np.where(self.labels==self.classes_[i])]
            tempData = tempData[:self.instances_[i]]
            
            tempLabel = self.labels[np.where(self.labels==self.classes_[i])]
            tempLabel = tempLabel[:self.instances_[i]]

            dataArray = np.vstack([dataArray, tempData]) if dataArray.size else tempData
            labelArray = np.hstack([labelArray, tempLabel]) if labelArray.size else tempLabel
            
            
        self.data = dataArray
        self.labels = labelArray
        
        # randomize the data
        p = np.random.permutation(self.data.shape[0])
        self.data = self.data[p]
        self.labels = self.labels[p]            
        
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, 
                 root, 
                 classes_, 
                 instances_,
                 train=True, 
                 transform=None, 
                 target_transform=None, 
                 download=False):
        
        self.classes_ = classes_
        self.instances_ = instances_
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set


        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        
        if self.train:
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
        
        self.data = self.data.numpy()      
        self.labels = self.labels.numpy()

        dataArray = np.array([])
        labelArray = np.array([])

        # randomize and get required number of instances
        for i in range(len(classes_)):

            tempData = self.data[np.where(self.labels==self.classes_[i])]
            tempData = tempData[:self.instances_[i]]

            tempLabel = self.labels[np.where(self.labels==self.classes_[i])]
            tempLabel = tempLabel[:self.instances_[i]]

            dataArray = np.vstack([dataArray, tempData]) if dataArray.size else tempData
            labelArray = np.hstack([labelArray, tempLabel]) if labelArray.size else tempLabel
        
        self.data = dataArray
        self.labels = labelArray 
        
        # randomize and get required number of instances
        p = np.random.permutation(self.data.shape[0])
        self.data = self.data[p]
        self.labels = self.labels[p]
    
          
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

# US Postal Service dataset
class USPS(data.Dataset):
    """USPS Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """


    def __init__(self, 
                 root,
                 classes_,
                 instances_,
                 train=True, 
                 transform=None,
                 target_transform=None):
        
        """Init USPS dataset."""
        
        # init params
        self.root = os.path.expanduser(root)
        self.train = train        
        self.filename = "usps_resampled.mat"
        self.classes_=classes_
        self.instances_ = instances_
        self.transform = transform
        self.target_transform = target_transform

        # Num of Train = 7438, Num ot Test 1860
        # Num of train = 4649 each
        
        
        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio
        
        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))
        
        if self.train:
            self.data = loaded_mat['train_patterns']
            self.labels = loaded_mat['train_labels']
            np.place(self.labels,self.labels==-1,0)
        else :
            self.data = loaded_mat['test_patterns']
            self.labels = loaded_mat['test_labels']
            np.place(self.labels,self.labels==-1,0)
        
        self.data = self.data.reshape(16,16,-1).transpose((2,0,1))
        self.data = ((self.data+1.0)*255.0)/2.0
        self.data = self.data.astype(dtype=np.uint8)
        self.labels = np.nonzero(self.labels.T)[1]

        
        dataArray = np.array([])
        labelArray = np.array([])
        
        # randomize and get required number of instances
        for i in range(len(classes_)):
           
            tempData = self.data[np.where(self.labels==self.classes_[i])]
            tempData = tempData[:self.instances_[i]]
            
            tempLabel = self.labels[np.where(self.labels==self.classes_[i])]
            tempLabel = tempLabel[:self.instances_[i]]

            dataArray = np.vstack([dataArray, tempData]) if dataArray.size else tempData
            labelArray = np.hstack([labelArray, tempLabel]) if labelArray.size else tempLabel
            
            
        self.data = dataArray
        self.labels = labelArray
        
        # randomize the data
        p = np.random.permutation(self.data.shape[0])
        self.data = self.data[p]
        self.labels = self.labels[p]  
        



    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

        #label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])

    def __len__(self):
        """Return size of dataset."""
        return self.data.shape[0]


    
    
class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

class CIFAR(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    
    train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_list = ['test_batch']

    def __init__(self, root, classes_, instances_, train=True,
                 transform=None, target_transform=None
                 ):
        
        self.classes_ = classes_
        self.instances_ = instances_
        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.data = []
            self.labels = []
            for fentry in self.train_list:
                
                f = fentry
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.data.append(entry['data'])
                
                if 'labels' in entry:
                    self.labels += entry['labels']
                else:
                    self.labels += entry['fine_labels']
                fo.close()

            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((50000, 3, 32, 32))             
        else:
            
            f = self.test_list[0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
                
            self.data = entry['data']
            if 'labels' in entry:
                self.labels = entry['labels']
            else:
                self.labels = entry['fine_labels']
            fo.close()
            
            self.data = self.data.reshape((10000, 3, 32, 32))
            
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.labels = np.asarray(self.labels)

        dataArray = np.array([])
        labelArray = np.array([])

        # randomize and get required number of instances
        for i in range(len(classes_)):

            tempData = self.data[np.where(self.labels==self.classes_[i])]
            tempData = tempData[:self.instances_[i]]

            tempLabel = self.labels[np.where(self.labels==self.classes_[i])]
            tempLabel = tempLabel[:self.instances_[i]]

            dataArray = np.vstack([dataArray, tempData]) if dataArray.size else tempData
            labelArray = np.hstack([labelArray, tempLabel]) if labelArray.size else tempLabel            

        self.data = dataArray
        self.labels = labelArray 

        # randomize and get required number of instances
        p = np.random.permutation(self.data.shape[0])
        self.data = self.data[p]
        self.labels = self.labels[p]   
            
           
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):

        return len(self.data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

# change this function
class EMNIST(MNIST):
    """`EMNIST <https://www.nist.gov/itl/iad/image-group/emnist-dataset/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    url = 'http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')

    def __init__(self, root, split, **kwargs):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        self.split = split
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super(EMNIST, self).__init__(root, **kwargs)

    def _training_file(self, split):
        return 'training_{}.pt'.format(split)

    def _test_file(self, split):
        return 'test_{}.pt'.format(split)

    def download(self):
        """Download the EMNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip
        import shutil
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url)
        data = urllib.request.urlopen(self.url)
        filename = self.url.rpartition('/')[2]
        raw_folder = os.path.join(self.root, self.raw_folder)
        file_path = os.path.join(raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print('Extracting zip archive')
        with zipfile.ZipFile(file_path) as zip_f:
            zip_f.extractall(raw_folder)
        os.unlink(file_path)
        gzip_folder = os.path.join(raw_folder, 'gzip')
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith('.gz'):
                print('Extracting ' + gzip_file)
                with open(os.path.join(raw_folder, gzip_file.replace('.gz', '')), 'wb') as out_f, \
                        gzip.GzipFile(os.path.join(gzip_folder, gzip_file)) as zip_f:
                    out_f.write(zip_f.read())
        shutil.rmtree(gzip_folder)

        # process and save as torch files
        for split in self.splits:
            print('Processing ' + split)
            training_set = (
                read_image_file(os.path.join(raw_folder, 'emnist-{}-train-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(raw_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(split)))
            )
            test_set = (
                read_image_file(os.path.join(raw_folder, 'emnist-{}-test-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(raw_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(split)))
            )
            with open(os.path.join(self.root, self.processed_folder, self._training_file(split)), 'wb') as f:
                torch.save(training_set, f)
            with open(os.path.join(self.root, self.processed_folder, self._test_file(split)), 'wb') as f:
                torch.save(test_set, f)

        print('Done!')
        
class Birds(data.Dataset):
    '''
    For CUBS and NABirds
    '''
    def __init__(self, root, instances_, split='train', transform=None, target_transform=None, bboxCrop=False):
        
        # know if it is a training or test set
        self.split = split
        
        # get information about the images
        imageDir ='images'
        
        # get bounding box information
        bbFile = 'bounding_boxes_'+ split +'.txt'
        nameFile = 'images_'+ split +'.txt'
        
        bbPath = os.path.join(root, bbFile)
        namePath = os.path.join(root, nameFile)
        imagePath = os.path.join(root, imageDir)

        file_df = pandas.read_csv(namePath, 
                          sep=' ', 
                          header=None, 
                          names = ["Index","FileName"])
        bb_df = pandas.read_csv(bbPath, 
                                sep=' ', 
                                header=None, 
                                names = ["Index","x","y","w","h"])

        df=[file_df, bb_df]
        
        # dataframe containing Index, FileName, x, y, width, height of bounding box
        df_final = reduce(lambda left,right: pandas.merge(left,right,on='Index'), df)
        
        self.root = root
        self.imagePath = imagePath
        
        # convey the bounding box information to make
        self.df_final = df_final

        self.transform = transform
        self.target_transform = target_transform
        
        # get the classes for all of the images
        classes, class_to_idx = self.find_classes()
        self.classes = classes
        self.num_classes = len(classes)
        self.class_to_idx = class_to_idx
        
        
        
        # get image paths
        imgs = self.make_dataset()
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        random.shuffle(imgs)
        
        self.imgs = imgs[:instances_]
        self.bboxCrop = bboxCrop
        
        print('num_classes', self.num_classes)

    def find_classes(self):
        
            dir = self.imagePath
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return classes, class_to_idx

    def make_dataset(self):
        
        '''
        Returns : List of images, tagets and bounding box attributes (only for training data)
        '''
        imagePath = self.imagePath
        df_final = self.df_final
        class_to_idx = self.class_to_idx
        
        images = []
        for key,values in df_final.iterrows():
            if is_image_file(os.path.join(imagePath, values[1])):
                item = (os.path.join(imagePath, values[1]), 
                        class_to_idx[values[1].split('/')[0]],
                        values[2],
                        values[3],
                        values[4],
                        values[5])
                images.append(item)
        print ('The number of images: ', len(images))
        return images

    def __getitem__(self, index):
        
        path, target, x, y, w, h  = self.imgs[index]
        
        # crop image here
        # return the PIL image
        image = get_imgs(path, bboxCrop=self.bboxCrop, bbox=(x,y,w,h))
        
        
        # apply transforms on the image
        if self.transform is not None:
            sample = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # return images in floatTensor and their targets
        return sample, target

    def __len__(self):
        return len(self.imgs)
    
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    