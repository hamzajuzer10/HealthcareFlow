import torch
from torch.utils.data import Dataset
import json
import os
import pandas as pd


class PathwayDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Load encoded pathways (completely into memory)
        with open(os.path.join(data_folder, self.split + '_PATHWAYS' + '.json'), 'r') as j:
            self.pathways = json.load(j)

        # Load pathway lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_PATHWAY_LEN' + '.json'), 'r') as j:
            self.pathway_len = json.load(j)

        # Load ward los (completely into memory)
        with open(os.path.join(data_folder, self.split + '_LOS' + '.json'), 'r') as j:
            self.los = json.load(j)

        # Load continuous features (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CONT_FEATURES' + '.json'), 'r') as j:
            self.cont_features = json.load(j)

        # Load categorical features (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAT_FEATURES' + '.json'), 'r') as j:
            self.cat_features = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.pathways)

    def __getitem__(self, i):

        cn_features = torch.FloatTensor(self.cont_features[i])

        ct_features = torch.LongTensor(self.cat_features[i])

        pathway = torch.LongTensor(self.pathways[i])

        p_len = torch.LongTensor([self.pathway_len[i]])

        los = torch.FloatTensor(self.los[i])

        return cn_features, ct_features, pathway, p_len, los

    def __len__(self):
        return self.dataset_size
