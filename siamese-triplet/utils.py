from itertools import combinations, product
__metaclass__ = type

import numpy as np
import torch



def pdist(vectors):
    
    '''
    Find the distance bwtween two vectors within a matrix
    '''
    
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

        
    def get_pairs(self, embeddings, labels):
        
        
        labels = labels.cpu().data.numpy()
        print labels
        
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        #all_pairs = torch.LongTensor(all_pairs)
       
        
        #print (labels[all_pairs[:, 0]].shape, labels[all_pairs[:, 1]]).nonzero().shape
        
        
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)].numpy()]

        return torch.from_numpy(positive_pairs), torch.from_numpy(negative_pairs)


class HardNegativePairSelector(PairSelector):
    
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    # where do you get these embedding from 
    def get_pairs(self, embeddings, labels):
        
        if self.cpu:
            embeddings = embeddings.cpu()
            
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        
        # should be combination of two different arrays
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs

    
class TripletSelector:
    
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    
    
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        
        if self.cpu:
            embeddings = embeddings.cpu()
            
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            
            label_mask = (labels == label)
            
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
                
            negative_indices = np.where(np.logical_not(label_mask))[0]
            
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)








# adding distance metric between two different indices
def pdist2(vectors1, vectors2):
    
    '''
    Find the distance bwtween two vectors within two matrices
    [ In case primary and helper dataset are different ]
    '''
        
    distance_matrix =  -2 * vectors1.mm(torch.t(vectors2)) + vectors1.pow(2).sum(dim=1).view(1, -1) + vectors2.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix    


# Hard Negative Mining in case of two different embeddings
class PairSelector2:
    
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddingsPrimary, labelsPrimary, embeddingsHelper, labelsHelper):
        raise NotImplementedError


class AllPositivePairSelector2(PairSelector2):
    
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    
    def __init__(self, balance=True):
        super(AllPositivePairSelector2, self).__init__()
        self.balance = balance

        
    def get_pairs(self, embeddingsPrimary, labelsPrimary, embeddingsHelper, labelsHelper):
        """
        No use of embeddings as we are selecting all positive pairs
        """
        labelsPrimary = labelsPrimary.cpu().data.numpy()
        labelsHelper = labelsHelper.cpu().data.numpy()
        
        # combine pairs in both the datasets
        
        # assertion that length of primary labels should be equal to length of helper labels
        all_pairs = np.array(list(combinations(range(len(labelsPrimary)), 2)))
        #all_pairs = torch.LongTensor(all_pairs)
       
        
        #print (labels[all_pairs[:, 0]].shape, labels[all_pairs[:, 1]]).nonzero().shape
        
        
        positive_pairs = all_pairs[(labelsPrimary[all_pairs[:, 0]] == labelsHelper[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labelsPrimary[all_pairs[:, 0]] != labelsHelper[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)].numpy()]

        return torch.from_numpy(positive_pairs), torch.from_numpy(negative_pairs)


class HardNegativePairSelector2(PairSelector2):
    
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector2, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddingsPrimary, labelsPrimary, embeddingsHelper, labelsHelper):
        if self.cpu:
            embeddingsPrimary = embeddingsPrimary.cpu()
            embeddingsHelper = embeddingsHelper.cpu()
            
        distance_matrix = pdist2(embeddingsPrimary, embeddingsHelper)

        labelsPrimary = labelsPrimary.cpu().data.numpy()
        labelsHelper = labelsHelper.cpu().data.numpy()
        
        # should be combination of two different arrays
        all_pairs = np.array(list(combinations(range(len(labelsPrimary)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labelsPrimary[all_pairs[:, 0]] == labelsHelper[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labelsPrimary[all_pairs[:, 0]] != labelsHelper[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs
    
    

class TripletSelector2:
    
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddingsPrimary, labelsPrimary, embeddingsHelper, labelsHelper):
        raise NotImplementedError


class AllTripletSelector2(TripletSelector):
    
    
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddingsPrimary, labelsPrimary, embeddingsHelper, labelsHelper):
        
        labelsPrimary = labelsPrimary.cpu().data.numpy()
        labelsHelper = labelsHelper.cpu().data.numpy()
        
        
        # get triplet corresponding to every label as anchor
        # TODO a dictionary which conveys the classes in helper which are near to the primary class
        
        triplets = []
        for label in set(labelsPrimary):
            
            # indices from helper domain which correspond to negative
            label_mask_primary = (labelsPrimary == label)
            label_indices_primary = np.where(label_mask_primary)[0]
            
            label_mask_helper = (labelsHelper == label)
            label_indices_helper = np.where(label_mask_helper)[0]
            
            
            # if only one primary index belonging to a particular class,
            # no other index can be found if it is selected as anchor,
            # hence need to continue
            if len(label_indices_primary) < 2:
                continue
            
            negative_indices_primary = np.where(np.logical_not(label_mask_primary))[0]
            negative_indices_helper = np.where(np.logical_not(label_mask_helper))[0]
            
            # a way to do this
            anchor_positives = list(product(label_indices_primary, label_indices_helper))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices_helper]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))



class FunctionNegativeTripletSelector2(TripletSelector):
    
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector2, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddingsPrimary, labelsPrimary, embeddingsHelper, labelsHelper):
        
        if self.cpu:
            embeddingsPrimary = embeddingsPrimary.cpu()
            embeddingsHelper = embeddingsHelper.cpu()
            
        distance_matrix = pdist2(embeddingsPrimary, embeddingsHelper)
        distance_matrix = distance_matrix.cpu()

        labelsPrimary = labelsPrimary.cpu().data.numpy()
        labelsHelper = labelsHelper.cpu().data.numpy()
        
        triplets = []

        for label in set(labelsPrimary):
            
            label_mask_primary = (labelsPrimary == label)
            label_mask_helper = (labelsHelper == label)
            
            label_indices_primary = np.where(label_mask_primary)[0]
            label_indices_helper = np.where(label_mask_helper)[0]
            
            if len(label_indices_primary) < 2:
                continue
                
                
            negative_indices_primary = np.where(np.logical_not(label_mask_primary))[0]
            negative_indices_helper = np.where(np.logical_not(label_mask_helper))[0]
            
            
            anchor_positives = list(product(label_indices_primary, label_indices_helper))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            # calculate the distance matrix b/n anchor and positive
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            
            
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices_helper)] + self.margin
                
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                
                if hard_negative is not None:
                    
                    # indices in helper domain which are hard to classify
                    hard_negative = negative_indices_helper[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
        
        # if there is no hard negative, select the first negative value
        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

def HardestNegativeTripletSelector2(margin, cpu=False): return FunctionNegativeTripletSelector2(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector2(margin, cpu=False): return FunctionNegativeTripletSelector2(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector2(margin, cpu=False): return FunctionNegativeTripletSelector2(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)

