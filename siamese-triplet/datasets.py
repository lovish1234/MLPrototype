import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

class SiameseCD(Dataset):
    
    
    def __init__(self, primaryDataset, helperDataset):
        self.primaryDataset = primaryDataset
        self.helperDataset = helperDataset

        self.primaryTrain = self.primaryDataset.train
        self.helperTrain = self.helperDataset.train
        # primaryTrain!=helperTrain then there is problem

        self.primaryTransform = self.primaryDataset.transform
        self.helperTransform = self.helperDataset.transform

        self.primaryLabels = self.primaryDataset.labels
        self.helperLabels = self.helperDataset.labels

        self.primaryData = self.primaryDataset.data
        self.helperData = self.helperDataset.data

        self.primaryLabelSet = set(self.primaryLabels)
        self.helperLabelSet = set(self.helperLabels)

        self.primaryLabelToIndices = {label: np.where(self.primaryLabels == label)[0]
                                          for label in self.primaryLabelSet}
        self.helperLabelToIndices = {label: np.where(self.helperLabels == label)[0]
                                          for label in self.helperLabelSet}

        if self.primaryTrain==False:

            # create fixed positive and negative pairs of images

            random_state = np.random.RandomState(29)

            # half of them should be positive pairs (let them be even)
            positive_pairs = [[i,
                               random_state.choice(self.helperLabelToIndices[self.primaryLabels[i]]),
                               1]
                              for i in range(0, len(self.primaryData), 2)]

            # half of them should be negative pairs (let them be even)
            negative_pairs = [[i,
                               random_state.choice(self.helperLabelToIndices[np.random.choice(
                                                           list(self.helperLabelSet - set([self.primaryLabels[i]])))]),
                               0]
                              for i in range(1, len(self.primaryData), 2)]
            self.pairs = positive_pairs + negative_pairs
        
        
        
    def __getitem__(self, index):
        
        if self.primaryTrain==True:
            
            pairClass = np.random.randint(0,2)
            primaryDomainImage, primaryDomainLabel = self.primaryData[index], self.primaryLabels[index]
            
            if pairClass==1:
                
                siameseIndex = index
                while siameseIndex== index:
                    siameseIndex = np.random.choice(self.helperLabelToIndices[primaryDomainLabel])

            else:
                # find another image with different index and different label
                # find the label first
                
                helperDomainLabel = np.random.choice(list(self.helperLabelSet-set([primaryDomainLabel])))
                siameseIndex = np.random.choice(self.helperLabelToIndices[helperDomainLabel])

            helperDomainImage = self.helperData[siameseIndex]  
            
        else:
            
            primaryDomainImage = self.primaryData[self.pairs[index][0]]
            helperDomainImage = self.helperData[self.pairs[index][1]]
            pairClass = self.pairs[index][2]
                
        # if the dataset has 1 channel or 3 channels !!!
        # MNIST is primary domain and SVHN is helper domain
        
        #print primaryDomainImage.shape[1]
        #print helperDomainImage.shape[1]
        
        if primaryDomainImage.shape[0]==3:
            primaryDomainImage = Image.fromarray(np.transpose(primaryDomainImage,(1,2,0)), mode='RGB')
        else :
            primaryDomainImage = Image.fromarray(primaryDomainImage, mode='L')
        
        if helperDomainImage.shape[0]==3:
            helperDomainImage = Image.fromarray(np.transpose(helperDomainImage,(1,2,0)), mode='RGB')
        else :
            helperDomainImage = Image.fromarray(helperDomainImage, mode='L')        

        if self.primaryTransform is not None :
            primaryDomainImage = self.primaryTransform(primaryDomainImage)
        
        
        if self.helperTransform is not None:
            helperDomainImage = self.helperTransform(helperDomainImage) 
        
        
        return (primaryDomainImage, helperDomainImage), pairClass  
        
    def __len__(self):
        return len(self.primaryDataset)    
    
    
    

class SiameseSVHN(Dataset):
    """
    Train : For each sample creates randomly a positive or negative pair
    Test : Creates a fixed pair for testing
    """
    
    def __init__(self, svhn_dataset):
        
        self.svhn_dataset = svhn_dataset
        
        self.train = self.svhn_dataset.train
        self.transform = self.svhn_dataset.transform
        self.labels = self.svhn_dataset.labels
        self.data = self.svhn_dataset.data
        
        # set of all the labels
        self.label_set = set(self.labels)

        # dict where all the label and keys and numpy arrays are values
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                      for label in self.label_set}
        
        if self.train==False:
            
            # create fixed positive and negative pairs of images

            random_state = np.random.RandomState(29)
            
            # half of them should be positive pairs (let them be even)
            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.labels[i]]),
                               1]
                              for i in range(0, len(self.data), 2)]
            
            # half of them should be negative pairs (let them be even)
            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[np.random.choice(
                                                           list(self.label_set - set([self.labels[i]])))]),
                               0]
                              for i in range(1, len(self.data), 2)]
            self.pairs = positive_pairs + negative_pairs
            
            
        
    def __getitem__(self, index):
        
        if self.train ==True:
            pairClass = np.random.randint(0,2)
            firstImage, firstLabel = self.data[index], self.labels[index]

            # if the pair is to be marked as positive
            if pairClass == 1:
                # find another image with a different index but same label
                siameseIndex = index
                while siameseIndex == index:
                    siameseIndex = np.random.choice(self.label_to_indices[firstLabel])
            else:
                # find another image with different index and different label

                # find the label first
                siameseLabel = np.random.choice(list(self.label_set-set([firstLabel])))
                siameseIndex = np.random.choice(self.label_to_indices[siameseLabel])
            secondImage = self.data[siameseIndex]
        else:
            
            # get the fixed lists [firstIndex, secondIndex, 1] and [firstIndex, secondIndex, 0]
            firstImage = self.data[self.pairs[index][0]]
            secondImage = self.data[self.pairs[index][1]]
            pairClass = self.pairs[index][2]
        

        
        firstImage = Image.fromarray(np.transpose(firstImage,(1,2,0)), mode='RGB')
        secondImage = Image.fromarray(np.transpose(secondImage,(1,2,0)), mode='RGB')
        
        
        if self.transform is not None:
            firstImage = self.transform(firstImage)
            secondImage = self.transform(secondImage) 
        
        return (firstImage, secondImage), pairClass
          
    def __len__(self):
        return len(self.svhn_dataset) 
    
    
    
class SiameseMNIST(Dataset):
    
    """
    Train : For each sample creates randomly a positive or negative pair
    Test : Creates a fixed pair for testing
    """
    
    def __init__(self, mnist_dataset):
        
        self.mnist_dataset = mnist_dataset
        
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform
        self.labels = self.mnist_dataset.labels
        self.data = self.mnist_dataset.data
        
        # set of all the labels
        self.label_set = set(self.labels)

        # dict where all the label and keys and numpy arrays are values
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                      for label in self.label_set}
        
        if self.train==False:
            
            # create fixed positive and negative pairs of images

            random_state = np.random.RandomState(29)
            
            # half of them should be positive pairs (let them be even)
            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.labels[i]]),
                               1]
                              for i in range(0, len(self.data), 2)]
            
            # half of them should be negative pairs (let them be even)
            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[np.random.choice(
                                                           list(self.label_set - set([self.labels[i]])))]),
                               0]
                              for i in range(1, len(self.data), 2)]
            self.pairs = positive_pairs + negative_pairs
            
            
        
    def __getitem__(self, index):
        
        if self.train ==True:
            pairClass = np.random.randint(0,2)
            firstImage, firstLabel = self.data[index], self.labels[index]

            # if the pair is to be marked as positive
            if pairClass == 1:
                # find another image with a different index but same label
                siameseIndex = index
                while siameseIndex == index:
                    siameseIndex = np.random.choice(self.label_to_indices[firstLabel])
            else:
                # find another image with different index and different label

                # find the label first
                siameseLabel = np.random.choice(list(self.label_set-set([firstLabel])))
                siameseIndex = np.random.choice(self.label_to_indices[siameseLabel])
            secondImage = self.data[siameseIndex]
        else:
            
            # get the fixed lists [firstIndex, secondIndex, 1] and [firstIndex, secondIndex, 0]
            firstImage = self.data[self.pairs[index][0]]
            secondImage = self.data[self.pairs[index][1]]
            pairClass = self.pairs[index][2]
        
        #print type(firstImage)
        #print firstImage.shape
        
        firstImage = Image.fromarray(firstImage, mode='L')
        secondImage = Image.fromarray(secondImage, mode='L')
        if self.transform is not None:
            firstImage = self.transform(firstImage)
            secondImage = self.transform(secondImage) 
        
        
        return (firstImage, secondImage), pairClass
          
    def __len__(self):
        return len(self.mnist_dataset)    
    
'''
class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i]]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)
'''

class TripletCD(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing    
    """
    
    def __init__(self, primaryDataset, helperDataset):

        self.primaryDataset = primaryDataset
        self.helperDataset = helperDataset

        self.primaryTrain = self.primaryDataset.train
        self.helperTrain = self.helperDataset.train
        # primaryTrain!=helperTrain then there is problem

        self.primaryTransform = self.primaryDataset.transform
        self.helperTransform = self.helperDataset.transform

        self.primaryLabels = self.primaryDataset.labels
        self.helperLabels = self.helperDataset.labels

        self.primaryData = self.primaryDataset.data
        self.helperData = self.helperDataset.data

        self.primaryLabelSet = set(self.primaryLabels)
        self.helperLabelSet = set(self.helperLabels)
        
        self.primaryLabelToIndices = {label: np.where(self.primaryLabels == label)[0]
                                          for label in self.primaryLabelSet}
        self.helperLabelToIndices = {label: np.where(self.helperLabels == label)[0]
                                          for label in self.helperLabelSet}
        
        if self.primaryTrain==False:
            # create fixed positive and negative pairs of images

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.helperLabelToIndices[self.primaryLabels[i]]),
                         random_state.choice(self.helperLabelToIndices[
                                                 np.random.choice(
                                                     list(self.helperLabelSet - set([self.primaryLabels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.primaryData))]
            
            self.triplets = triplets             

    
    def __getitem__(self, index):
        
        if self.primaryTrain:
            
            primaryDomainImage, primaryDomainLabel = self.primaryData[index], self.primaryLabels[index]
            
            
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.helperLabelToIndices[primaryDomainLabel])
                
            negative_label = np.random.choice(list(self.helperLabelSet - set([primaryDomainLabel])))
            negative_index = np.random.choice(self.helperLabelToIndices[negative_label])
            
            helperDomainPositiveImage = self.helperData[positive_index]
            helperDomainNegativeImage = self.helperData[negative_index]
            
        else:
            
            primaryDomainImage = self.primaryData[self.triplets[index][0]]
            helperDomainPositiveImage = self.helperData[self.triplets[index][1]]
            helperDomainNegativeImage = self.helperData[self.triplets[index][2]]
        
        if primaryDomainImage.shape[0]==3:
            primaryDomainImage = Image.fromarray(np.transpose(primaryDomainImage,(1,2,0)), mode='RGB')
        else :
            primaryDomainImage = Image.fromarray(primaryDomainImage, mode='L')
        
        if helperDomainPositiveImage.shape[0]==3:
            helperDomainPositiveImage = Image.fromarray(np.transpose(helperDomainPositiveImage,(1,2,0)), mode='RGB')
        else :
            helperDomainPositiveImage = Image.fromarray(helperDomainPositiveImage, mode='L')
            
        if helperDomainNegativeImage.shape[0]==3:
            helperDomainNegativeImage = Image.fromarray(np.transpose(helperDomainNegativeImage,(1,2,0)), mode='RGB')
        else :
            helperDomainNegativeImage = Image.fromarray(helperDomainNegativeImage, mode='L')
        
        if self.primaryTransform is not None:
            primaryDomainImage = self.primaryTransform(primaryDomainImage)
        
        if self.helperTransform is not None:
            helperDomainPositiveImage = self.helperTransform(helperDomainPositiveImage)
            helperDomainNegativeImage = self.helperTransform(helperDomainNegativeImage)
        return (primaryDomainImage, helperDomainPositiveImage,  helperDomainNegativeImage), []
    
    
        
        
    def __len__(self):
        return len(self.primaryDataset)   
            
        

class TripletSVHN(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing    
    """
    
    
    def __init__(self, svhn_dataset):
        
        self.svhn_dataset = svhn_dataset
        
        self.train = self.svhn_dataset.train
        self.transform = self.svhn_dataset.transform
        self.labels = self.svhn_dataset.labels
        self.data = self.svhn_dataset.data
        
        # set of all the labels
        self.label_set = set(self.labels)

        # dict where all the label and keys and numpy arrays are values
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                      for label in self.label_set}
        
        # change the code below
        if self.train==False:
            
            # create fixed positive and negative pairs of images

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.label_set - set([self.labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.data))]
            
            self.triplets = triplets            
       

    def __getitem__(self, index):
        
        
        if self.train:
            
            firstImage, firstLabel = self.data[index], self.labels[index]
            
            
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[firstLabel])
                
            negative_label = np.random.choice(list(self.label_set - set([firstLabel])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            
            secondImage = self.data[positive_index]
            thirdImage = self.data[negative_index]
            
        else:
            
            firstImage = self.data[self.triplets[index][0]]
            secondImage = self.data[self.triplets[index][1]]
            thirdImage = self.data[self.triplets[index][2]]

        firstImage = Image.fromarray(np.transpose(firstImage,(1,2,0)), mode='RGB')
        secondImage = Image.fromarray(np.transpose(secondImage,(1,2,0)), mode='RGB')
        thirdImage = Image.fromarray(np.transpose(thirdImage,(1,2,0)), mode='RGB')
        
        if self.transform is not None:
            firstImage = self.transform(firstImage)
            secondImage = self.transform(secondImage)
            thirdImage = self.transform(thirdImage)
        return (firstImage, secondImage, thirdImage), []

    def __len__(self):
        return len(self.svhn_dataset)        
    
    
class TripletMNIST(Dataset):
    
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, mnist_dataset):
        
        self.mnist_dataset = mnist_dataset
        
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform
        self.labels = self.mnist_dataset.labels
        self.data = self.mnist_dataset.data
        
        # set of all the labels
        self.label_set = set(self.labels)

        # dict where all the label and keys and numpy arrays are values
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                      for label in self.label_set}
        
        # change the code below
        if self.train==False:
            
            # create fixed positive and negative pairs of images

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.label_set - set([self.labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.data))]
            
            self.triplets = triplets            
       
    def __getitem__(self, index):
        
        
        if self.train:
            
            firstImage, firstLabel = self.data[index], self.labels[index]
            
            
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[firstLabel])
                
            negative_label = np.random.choice(list(self.label_set - set([firstLabel])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            
            secondImage = self.data[positive_index]
            thirdImage = self.data[negative_index]
            
        else:
            
            firstImage = self.data[self.triplets[index][0]]
            secondImage = self.data[self.triplets[index][1]]
            thirdImage = self.data[self.triplets[index][2]]

        firstImage = Image.fromarray(firstImage, mode='L')
        secondImage = Image.fromarray(secondImage, mode='L')
        thirdImage = Image.fromarray(thirdImage, mode='L')
        
        if self.transform is not None:
            firstImage = self.transform(firstImage)
            secondImage = self.transform(secondImage)
            thirdImage = self.transform(thirdImage)
        return (firstImage, secondImage, thirdImage), []

    def __len__(self):
        return len(self.mnist_dataset)   
    
    
'''    

class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)

'''


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
