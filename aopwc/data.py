import os
import torch
import numpy as np
from torch.utils.data import Dataset


# Statistics calculated over training set [0:40000] of phase_screens_part1 
# see scripts/normalize_phase_screens_v2.py for more details
WAVEFRONT_MEAN = -9.59
WAVEFRONT_STD = 2161.88


class WavefrontDataset(Dataset):
    
    """
    Dataset class for loading wavefront sequences

    Args:
        directory (str): path to folder containing npy files
    """

    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        self.filenames = sorted(os.listdir(directory))
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError()
        
        path = os.path.join(self.directory, self.filenames[index])
        return torch.from_numpy(np.load(path).astype(np.float32))


class TipTiltDataset(Dataset):
    """
    Dataset class for loading open loop tip-tilt data

    Args:
        directory (str): path to folder containing npy files
    """

    def __init__(self, directory):
        examples = list()

        # Pre-load examples into memory
        for filename in sorted(os.listdir(directory)):
            path = os.path.join(directory, filename)
            data = torch.from_numpy(np.load(path).astype(np.float32))
            examples.append(data.transpose(0, 1))
        
        # Concatenate into a single torch tensor
        self.examples = torch.stack(examples)
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]



class SubDataset(Dataset):
    
    """
    A dataset class representing a subset of another dataset

    Args:
        dataset (Dataset): the parent dataset
        indices (list or np.ndarray): an array of indices representing the 
            subset of the parent dataset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.array(indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError()
        return self.dataset[self.indices[index]]


def split_dataset(dataset, split_sizes, shuffle=False):
    
    """
    Splits a dataset into subsets of the specified size.

    Args:
        dataset (Dataset): the dataset to split
        split_sizes (tuple of floats): iterable of floats in [0, 1] representing
            what proportion of the dataset should be assigned to each subset
        shuffle (bool): randomly shuffle the dataset when splitting

    Returns:
        splits (tuple of SubDataset): a tuple of datasets containing each 
            subset of the data 
    """

    if shuffle:
        # Generate a random permutation of the dataset indices
        indices = np.random.randperm(len(dataset))
    else:
        indices = np.arange(len(dataset))
    
    # Split index array into chunks of the specified size
    split_sizes = (np.array(split_sizes).cumsum() * len(dataset)).astype(int)
    split_inds = np.split(indices, split_sizes)[:len(split_sizes)]

    # Return subset datasets
    return [SubDataset(dataset, inds) for inds in split_inds]
