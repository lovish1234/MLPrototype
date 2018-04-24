# dataset directory
rootDir ='/Neutron3/lovish.chum/'
resultDir='/Neutron3/lovish.chum/'

modelSubDir = 'models/'
plotSubDir = 'plots/'
resultSubDir = 'results/'

nonMMDSubDir = 'nonMMD/'
MMDSubDir = 'MMDall/'
crossMMDSubDir = 'crossMMDall/'

samplesSubDir = 'samples/'
compressedSubDir = 'compressed/'


# set to True if running the code on a GPU
cuda=True

# noise size for both generator and discriminator
numInputChannels=100

# number of channels in generator and discriminator
numGenFilter=128
numDiscFilter=128

# number of images in a single batch passing through DCGAN
batchSize = 128
lambdaMMD = 1

# parameters for adam optimizer
beta1 = 0.5
beta2 = 0.999
optimBetas = (beta1, beta2)
learningRate = 0.0002

# display image inside the ipython notebook
showImage = 1

# number of samples to be produced while testing
numOfSamples=2000