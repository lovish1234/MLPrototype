import torch
#from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator, Discriminator
import utils
import argparse
import os, itertools
import numpy as np

# parse the data in MNIST format
import struct
import random


# resize the image to the one desired
# by input
from skimage.transform import resize



parser = argparse.ArgumentParser()
parser.add_argument('--cuda', required=False, default=True, type=bool, help='gpu availability')
parser.add_argument('--log', required=False, default=True, type=bool, help='tensorboard logging ')

parser.add_argument('--dataset', required=False, default='MNIST', help='input dataset')
parser.add_argument('--batch_size', type=int, default=100, help='train batch size')

# number of channels in input/output eg. Greyscale,1 or RGB,3 or RGBD,4
parser.add_argument('--input_channels', type=int, default=1, help='number of input channels')
parser.add_argument('--output_channels', type=int, default=1, help='number of output channels')

# number of filters in the first conv layer of  generator and discriminator
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=64)

# arguments for architecture
parser.add_argument('--num_resnet', type=int, default=6, help='number of resnet blocks in generator')
#parser.add_argument('--input_size', type=int, default=256, help='input size')

# arguments for changing the input 
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=64, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=False, help='random fliplr True of False')

# arguments for Loss function ( Mean Square
# and Cyclic )
parser.add_argument('--num_epochs', type=int, default=200, help='number of train epochs')
parser.add_argument('--decay_epoch', type=int, default=100, help='start decaying learning rate after this number')

# importance to the cyclic consistency loss
parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')
parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')

# adam optimiser parameters
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')

# data corresponding to source and target domain
# number of instances in source and target
parser.add_argument('--train_A', type=int, default=5, help='training data for domain A')
parser.add_argument('--train_B', type=int, default=5, help='training data for domain B')

# class ID for source and target domain 
parser.add_argument('--domain_A', type=int, default=1, help='domain data for domain A')
parser.add_argument('--domain_B', type=int, default=7, help='domain data for domain B')



params = parser.parse_args()
print(params)


# logging the error 
if params.log:
    print (params.log)
    
    from logger import Logger



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

'''
# Data pre-processing
transform = transforms.Compose([transforms.Scale(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
'''

# Train data
'''
train_data_A = DatasetFromFolder(data_dir, subfolder='trainA', transform=transform,
                                 resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
train_data_loader_A = torch.utils.data.DataLoader(dataset=train_data_A,
                                                  batch_size=params.batch_size,
                                                  shuffle=True)
'''

train_data_loader_A,train_data_A = loadDataset(params.dataset,params.domain_A,
                                               params.train_A,params.crop_size,params.fliplr)

'''
train_data_B = DatasetFromFolder(data_dir, subfolder='trainB', transform=transform,
                                 resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
train_data_loader_B = torch.utils.data.DataLoader(dataset=train_data_B,
                                                  batch_size=params.batch_size,
                                                  shuffle=True)
'''
train_data_loader_B,train_data_B = loadDataset(params.dataset,params.domain_B,
                                               params.train_B,params.crop_size,params.fliplr)

# Test data
'''
test_data_A = DatasetFromFolder(data_dir, subfolder='testA', transform=transform)
test_data_loader_A = torch.utils.data.DataLoader(dataset=test_data_A,
                                                 batch_size=params.batch_size,
                                                 shuffle=params.fliplr)
'''
test_data_loader_A,test_data_A = loadDataset(params.dataset,params.domain_A,
                                             10,params.crop_size,params.fliplr)


'''
test_data_B = DatasetFromFolder(data_dir, subfolder='testB', transform=transform)
test_data_loader_B = torch.utils.data.DataLoader(dataset=test_data_B,
                                                 batch_size=params.batch_size,
                                                 shuffle=params.fliplr)
'''
test_data_loader_B,test_data_B = loadDataset(params.dataset,params.domain_B,
                                             10,params.crop_size,params.fliplr)

# Get specific test images
test_real_A_data = test_data_A.__getitem__(9)[0].unsqueeze(0)  # Convert to 4d tensor (BxNxHxW)
test_real_B_data = test_data_B.__getitem__(7)[0].unsqueeze(0)

# Model Initialisation
G_A = Generator(params.input_channels, params.ngf, params.output_channels, params.num_resnet)
G_A.double()

G_B = Generator(params.input_channels, params.ngf, params.output_channels, params.num_resnet)
G_B.double()

D_A = Discriminator(params.input_channels, params.ndf, params.output_channels)
D_A.double()

D_B = Discriminator(params.input_channels, params.ndf, params.output_channels)
D_B.double()

# Weight Initialisation
G_A.normal_weight_init(mean=0.0, std=0.02)
G_B.normal_weight_init(mean=0.0, std=0.02)
D_A.normal_weight_init(mean=0.0, std=0.02)
D_B.normal_weight_init(mean=0.0, std=0.02)

G_A = gpuAvailable(G_A, params.cuda)
G_B = gpuAvailable(G_B, params.cuda)
D_A = gpuAvailable(D_A, params.cuda)
D_B = gpuAvailable(D_B, params.cuda)



if params.log:
    # Set the logger
    D_A_log_dir = save_dir + 'D_A_logs'
    D_B_log_dir = save_dir + 'D_B_logs'
    if not os.path.exists(D_A_log_dir):
        os.mkdir(D_A_log_dir)
    D_A_logger = Logger(D_A_log_dir)
    if not os.path.exists(D_B_log_dir):
        os.mkdir(D_B_log_dir)
    D_B_logger = Logger(D_B_log_dir)

    G_A_log_dir = save_dir + 'G_A_logs'
    G_B_log_dir = save_dir + 'G_B_logs'
    if not os.path.exists(G_A_log_dir):
        os.mkdir(G_A_log_dir)
    G_A_logger = Logger(G_A_log_dir)
    if not os.path.exists(G_B_log_dir):
        os.mkdir(G_B_log_dir)
    G_B_logger = Logger(G_B_log_dir)

    cycle_A_log_dir = save_dir + 'cycle_A_logs'
    cycle_B_log_dir = save_dir + 'cycle_B_logs'
    if not os.path.exists(cycle_A_log_dir):
        os.mkdir(cycle_A_log_dir)
    cycle_A_logger = Logger(cycle_A_log_dir)
    if not os.path.exists(cycle_B_log_dir):
        os.mkdir(cycle_B_log_dir)
    cycle_B_logger = Logger(cycle_B_log_dir)

    img_log_dir = save_dir + 'img_logs'
    if not os.path.exists(img_log_dir):
        os.mkdir(img_log_dir)
    img_logger = Logger(img_log_dir)


# Loss function
MSE_loss = gpuAvailable(torch.nn.MSELoss(), params.cuda)
L1_loss = gpuAvailable(torch.nn.L1Loss(), params.cuda)


# optimizers
G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=params.lrG, betas=(params.beta1, params.beta2))
D_A_optimizer = torch.optim.Adam(D_A.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))
D_B_optimizer = torch.optim.Adam(D_B.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

# Training GAN
D_A_avg_losses = []
D_B_avg_losses = []
G_A_avg_losses = []
G_B_avg_losses = []
cycle_A_avg_losses = []
cycle_B_avg_losses = []

# Generated image pool
num_pool = 50
fake_A_pool = utils.ImagePool(num_pool)
fake_B_pool = utils.ImagePool(num_pool)

step = 0
for epoch in range(params.num_epochs):
    D_A_losses = []
    D_B_losses = []
    G_A_losses = []
    G_B_losses = []
    cycle_A_losses = []
    cycle_B_losses = []

    # learning rate decay
    if (epoch + 1) > params.decay_epoch:
        D_A_optimizer.param_groups[0]['lr'] -= params.lrD / (params.num_epochs - params.decay_epoch)
        D_B_optimizer.param_groups[0]['lr'] -= params.lrD / (params.num_epochs - params.decay_epoch)
        G_optimizer.param_groups[0]['lr'] -= params.lrG / (params.num_epochs - params.decay_epoch)

    # training
    for i, (real_A, real_B) in enumerate(zip(train_data_loader_A, train_data_loader_B)):

        real_A = real_A[0]
        real_B = real_B[0]

        # input image data
        real_A = Variable(gpuAvailable(real_A, params.cuda))
        real_B = Variable(gpuAvailable(real_B, params.cuda))

        # Train generator G
        # A -> B
        fake_B = G_A(real_A)
        D_B_fake_decision = D_B(fake_B)
        G_A_loss = MSE_loss(D_B_fake_decision,
                Variable(gpuAvailable(torch.ones(D_B_fake_decision.size()).double(),
                    params.cuda)))

        # forward cycle loss
        recon_A = G_B(fake_B)
        cycle_A_loss = L1_loss(recon_A, real_A) * params.lambdaA

        # B -> A
        fake_A = G_B(real_B)
        D_A_fake_decision = D_A(fake_A)
        G_B_loss = MSE_loss(D_A_fake_decision,
                Variable(gpuAvailable(torch.ones(D_A_fake_decision.size()).double(),
                    params.cuda)))

        # backward cycle loss
        recon_B = G_A(fake_A)
        cycle_B_loss = L1_loss(recon_B, real_B) * params.lambdaB

        # Back propagation
        G_loss = G_A_loss + G_B_loss + cycle_A_loss + cycle_B_loss
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # Train discriminator D_A
        D_A_real_decision = D_A(real_A)
        D_A_real_loss = MSE_loss(D_A_real_decision,
                Variable(gpuAvailable(torch.ones(D_A_real_decision.size()).double(),
                    params.cuda)))
        fake_A = fake_A_pool.query(fake_A)
        D_A_fake_decision = D_A(fake_A)
        D_A_fake_loss = MSE_loss(D_A_fake_decision,
                Variable(gpuAvailable(torch.zeros(D_A_fake_decision.size()).double(),
                    params.cuda)))

        # Back propagation
        D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
        D_A_optimizer.zero_grad()
        D_A_loss.backward()
        D_A_optimizer.step()

        # Train discriminator D_B
        D_B_real_decision = D_B(real_B)
        D_B_real_loss = MSE_loss(D_B_real_decision,
                Variable(gpuAvailable(torch.ones(D_B_real_decision.size()).double(),
                    params.cuda)))
        fake_B = fake_B_pool.query(fake_B)
        D_B_fake_decision = D_B(fake_B)
        D_B_fake_loss = MSE_loss(D_B_fake_decision,
                Variable(gpuAvailable(torch.zeros(D_B_fake_decision.size()).double(),
                    params.cuda)))

        # Back propagation
        D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
        D_B_optimizer.zero_grad()
        D_B_loss.backward()
        D_B_optimizer.step()

        # loss values
        D_A_losses.append(D_A_loss.data[0])
        D_B_losses.append(D_B_loss.data[0])
        G_A_losses.append(G_A_loss.data[0])
        G_B_losses.append(G_B_loss.data[0])
        cycle_A_losses.append(cycle_A_loss.data[0])
        cycle_B_losses.append(cycle_B_loss.data[0])

        print('Epoch [%d/%d], Step [%d/%d], D_A_loss: %.4f, D_B_loss: %.4f, G_A_loss: %.4f, G_B_loss: %.4f'
              % (epoch+1, params.num_epochs, i+1, len(train_data_loader_A), D_A_loss.data[0], D_B_loss.data[0], G_A_loss.data[0], G_B_loss.data[0]))

        # ============ TensorBoard logging ============#
        if params.log:
            D_A_logger.scalar_summary('losses', D_A_loss.data[0], step + 1)
            D_B_logger.scalar_summary('losses', D_B_loss.data[0], step + 1)
            G_A_logger.scalar_summary('losses', G_A_loss.data[0], step + 1)
            G_B_logger.scalar_summary('losses', G_B_loss.data[0], step + 1)
            cycle_A_logger.scalar_summary('losses', cycle_A_loss.data[0], step + 1)
            cycle_B_logger.scalar_summary('losses', cycle_B_loss.data[0], step + 1)
            step += 1

    D_A_avg_loss = torch.mean(torch.FloatTensor(D_A_losses))
    D_B_avg_loss = torch.mean(torch.FloatTensor(D_B_losses))
    G_A_avg_loss = torch.mean(torch.FloatTensor(G_A_losses))
    G_B_avg_loss = torch.mean(torch.FloatTensor(G_B_losses))
    cycle_A_avg_loss = torch.mean(torch.FloatTensor(cycle_A_losses))
    cycle_B_avg_loss = torch.mean(torch.FloatTensor(cycle_B_losses))

    # avg loss values for plot
    D_A_avg_losses.append(D_A_avg_loss)
    D_B_avg_losses.append(D_B_avg_loss)
    G_A_avg_losses.append(G_A_avg_loss)
    G_B_avg_losses.append(G_B_avg_loss)
    cycle_A_avg_losses.append(cycle_A_avg_loss)
    cycle_B_avg_losses.append(cycle_B_avg_loss)

    # Show result for test image
    test_real_A = Variable(gpuAvailable(test_real_A_data, params.cuda))
    test_fake_B = G_A(test_real_A)
    test_recon_A = G_B(test_fake_B)

    test_real_B = Variable(gpuAvailable(test_real_B_data, params.cuda))
    test_fake_A = G_B(test_real_B)
    test_recon_B = G_A(test_fake_A)

    '''
    print (test_real_A.data.shape, test_real_B.data.shape)
    print (test_fake_B.data.shape, test_fake_A.data.shape)
    print (test_recon_A.data.shape, test_recon_B.data.shape)
    '''

    utils.plot_train_result([test_real_A, test_real_B],
                            [test_fake_B, test_fake_A],
                            [test_recon_A,test_recon_B],
                            epoch, save=True, save_dir=save_dir)

    # log the images
    result_AtoB = np.concatenate((utils.to_np(test_real_A), utils.to_np(test_fake_B), utils.to_np(test_recon_A)), axis=3)
    result_BtoA = np.concatenate((utils.to_np(test_real_B), utils.to_np(test_fake_A), utils.to_np(test_recon_B)), axis=3)

    info = {
        'result_AtoB': result_AtoB.transpose(0, 2, 3, 1),  # convert to BxHxWxC
        'result_BtoA': result_BtoA.transpose(0, 2, 3, 1)
    }

    if params.log:
        for tag, images in info.items(): 
            img_logger.image_summary(tag, images, epoch + 1)


# Plot average losses
avg_losses = []
avg_losses.append(D_A_avg_losses)
avg_losses.append(D_B_avg_losses)
avg_losses.append(G_A_avg_losses)
avg_losses.append(G_B_avg_losses)
avg_losses.append(cycle_A_avg_losses)
avg_losses.append(cycle_B_avg_losses)
utils.plot_loss(avg_losses, params.num_epochs, save=True, save_dir=save_dir)

# Make gif
utils.make_gif(params.dataset, params.num_epochs, save_dir=save_dir)
# Save trained parameters of model
torch.save(G_A.state_dict(), model_dir + 'generator_A_param.pkl')
torch.save(G_B.state_dict(), model_dir + 'generator_B_param.pkl')
torch.save(D_A.state_dict(), model_dir + 'discriminator_A_param.pkl')
torch.save(D_B.state_dict(), model_dir + 'discriminator_B_param.pkl')
