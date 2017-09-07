import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

class DataDistribution(object):
    '''
    The distribution which we would like to replicate.
    : A gaussian
    '''
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        #samples.sort()
        return samples

class GeneratorDistribution(object):
    '''
    Initial distribution of Generator initialised with
    stratified sampling
    '''
    def __init__(self, range):
        self.range = range

    def sample(self,N):
        return np.linspace(-self.range,self.range,N) + np.random.random(N)*0.01

def linear(input, outputDimension, scope=None, stdDev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
                'w',
                [input.get_shape()[1],outputDimension],
                initializer = tf.random_normal_initializer(stddev=stdDev)
                )
        b = tf.get_variable(
                'b',
                [outputDimension],
                initializer = tf.random_normal_initializer(stddev=stdDev)
                )
        return tf.matmul(input,w)+b

def generator(input, hiddenSize):
    '''
    Generator : A Neural Network with two hidden layers and 
    a non-linearity in the first layer.
    '''
    h0 = tf.nn.softplus(linear(input, hiddenSize, 'g0'))
    h1 = linear(h0,1,'g1')
    return h1

def discriminator(input, hiddenSize):
    '''
    Discriminator: Needs to be more powerful that generator.
    Hence has more layers with non-linearity.
    '''
    h0 = tf.nn.relu(linear(input,hiddenSize*2,'d0'))
    h1 = tf.nn.relu(linear(h0,hiddenSize*2,'d1'))
    h2 = tf.nn.relu(linear(h1,hiddenSize*2,'d2'))
    h3 = tf.sigmoid(linear(h2,1,'d3'))
    return h3

def optimizer(lossValue, variableList):
    learningRate = 0.001

    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(
            lossValue,
            global_step = step,
            var_list = variableList
            )
    return optimizer 

def log(x):
    return tf.log(tf.maximum(x,1e-5))

class GAN(object):
    def __init__(self,parameters):

        with tf.variable_scope('G'):
            # Generated Samples
            self.z = tf.placeholder(tf.float32, shape=(parameters.batchSize,1))
            self.G = generator(self.z, parameters.hiddenSize)
        with tf.variable_scope('D'):
            self.x = tf.placeholder(tf.float32, shape=(parameters.batchSize,1))
            self.D1 = discriminator(self.x, parameters.hiddenSize)
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(self.G, parameters.hiddenSize)


        self.lossDiscriminator = tf.reduce_mean(-log(self.D1) - log(1-self.D2))
        self.lossGenerator = tf.reduce_mean(-log(self.D2))
        
        variables = tf.trainable_variables()
        self.dParams = [ v for v in variables if v.name.startswith('D/')]
        self.gParams = [ v for v in variables if v.name.startswith('G/')]

        self.dOptimizer = optimizer(self.lossDiscriminator, self.dParams)
        self.gOptimizer = optimizer(self.lossGenerator, self.gParams)

def train(model, dataDistribution, generatorDistribution, parameters):

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(parameters.numOfSteps + 1):
            # the distribution we would like to replicate
            x = dataDistribution.sample(parameters.batchSize)

            # the distribution/noise we feed to generator initially
            z = generatorDistribution.sample(parameters.batchSize)

            #calculate discriminator loss
            lossDiscriminator,_, = session.run([model.lossDiscriminator,
                model.dOptimizer],
                {model.x: np.reshape(x,(parameters.batchSize,1)),
                 model.z: np.reshape(z,(parameters.batchSize,1))
                        })
            lossGenerator,_, = session.run([model.lossGenerator,
                model.gOptimizer],
                {model.z: np.reshape(z,(parameters.batchSize,1))
                    })
            if step % parameters.storeLog == 0:
                print('{}:{:.4f}\t{:.4f}'.format(step,lossDiscriminator,lossGenerator))
            
        # plotting functions
        samp = samples(model, session, dataDistribution,
                generatorDistribution.range, parameters.batchSize)
        plot_distributions(samp, generatorDistribution.range)

def samples(model, session, dataDistribution, generatorRange, batchSize,
        numOfPoints=10000, numOfBins=100):
    # Inference
    # Return the histogram with bins
    bins = np.linspace(-generatorRange,generatorRange,numOfBins)

    # Get the histogram of samples for data distribution
    d = dataDistribution.sample(numOfPoints)
    pd,_=np.histogram(d,bins=bins,density=True)

    # Get the histogram of generated distribution 
    zs = np.linspace(-generatorRange,generatorRange,numOfPoints)
    g = np.zeros((numOfPoints,1))

    for i in range(numOfPoints // batchSize):
        g[batchSize*i:batchSize*(i+1)] = session.run( 
                model.G,
                {
                    model.z: np.reshape(zs[batchSize*i:batchSize*(i+1)],
                        (batchSize,1)
                        )
                    }
                )
    pg,_ = np.histogram(g,bins=bins,density=True)

    xs = np.linspace(-generatorRange,generatorRange,numOfPoints)
    db = np.zeros((numOfPoints,1))
    for i in range(numOfPoints // batchSize):
        db[batchSize*i:batchSize*(i+1)] = session.run( 
                model.D1,
                {
                    model.x: np.reshape(xs[batchSize*i:batchSize*(i+1)],
                        (batchSize,1)
                        )
                    }
                )

    return db,pd,pg

def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()


def main(args):
    model = GAN(args)
    train(model, DataDistribution(), GeneratorDistribution(range=8), args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numOfSteps', type=int, default=5000,
                        help='the number of training steps to take')
    parser.add_argument('--hiddenSize', type=int, default=4,
                        help='MLP hidden size')
    parser.add_argument('--batchSize', type=int, default=8,
                        help='the batch size')
    parser.add_argument('--storeLog', type=int, default=10,
                        help='print loss after this many steps')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
