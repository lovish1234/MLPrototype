import torch.nn as nn
import torch.nn.functional as F

# changed this function
class EmbeddingNet(nn.Module):
    def __init__(self, noInputChannels):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(noInputChannels, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 5 * 5, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
    
 
class LeNet(nn.Module):
    def __init__(self, noInputChannels):
        super(LeNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(noInputChannels, 6, 5), nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(6, 16, 5), nn.ReLU(),
                                     nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(16*5*5,120), nn.ReLU(),
                                nn.Linear(120,84)) 
    def forward(self, x):
        
        output = self.convnet(x)
        output = output.view(output.size()[0],-1)
        output = self.fc(output)
        return output
    
    def get_embedding(self, x):
        return self.forward(x)
    


    
# embedding net with L2 norm 
class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)

# 
class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        #scores = F.log_softmax(self.fc1(output), dim=-1)
        scores = F.log_softmax(self.fc1(output))
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))

# changed this part
class SiameseNet(nn.Module):
    def __init__(self, embedding_net_1, embedding_net_2):
        super(SiameseNet, self).__init__()
        self.embedding_net_1 = embedding_net_1
        self.embedding_net_2 = embedding_net_2

    def forward(self, x1, x2):
        
        
        output1 = self.embedding_net_1(x1)
        output2 = self.embedding_net_2(x2)
        
        '''
        if x1.data.shape[1]==1:
            output1 = self.embedding_net_1(x1)
        else:
            output1 = self.embedding_net_2(x1)
            
        if x2.data.shape[1]==1:
            output2 = self.embedding_net_1(x2)
        else:
            output2 = self.embedding_net_2(x2)
        '''
        
        return output1, output2

    def get_embedding(self, x, domain='primary'):
        
        if domain=='primary':
            return self.embedding_net_1(x)
        else:
            return self.embedding_net_2(x)
        
        '''
        if x.data.shape[1]==1:
            return self.embedding_net_1(x)
        else:
            return self.embedding_net_2(x)
        '''

class TripletNet(nn.Module):
    def __init__(self, embedding_net_1, embedding_net_2):
        super(TripletNet, self).__init__()
        self.embedding_net_1 = embedding_net_1
        self.embedding_net_2 = embedding_net_2

    def forward(self, x1, x2, x3):

        
        output1 = self.embedding_net_1(x1)
        output2 = self.embedding_net_2(x2)
        output3 = self.embedding_net_2(x3)
        
        '''
        if x1.data.shape[1]==1:
            output1 = self.embedding_net_1(x1)
        else:
            output1 = self.embedding_net_2(x1)
            
        if x2.data.shape[1]==1:
            output2 = self.embedding_net_1(x2)
        else:
            output2 = self.embedding_net_2(x2)
        
        if x3.data.shape[1]==1:
            output3 = self.embedding_net_1(x3)
        else:
            output3 = self.embedding_net_2(x3)
        '''
        
        return output1, output2, output3

    def get_embedding(self, x, domain='primary'):
        
        if domain=='primary':
            return self.embedding_net_1(x)
        else:
            return self.embedding_net_2(x)
        
        '''
        if x.data.shape[1]==1:
            return self.embedding_net_1(x)
        else:
            return self.embedding_net_2(x)
        '''
