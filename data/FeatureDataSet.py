import torch
import numpy as np
from torch.utils.data import Dataset


# from torch.utils.data.sampler import BatchSampler


class FeatureDataSet(Dataset):

    def __init__(self, t_file, i_file):
        print("Using FeatureDataset WWW2022")
        test_t = np.load(t_file)
        self.test_data_t = torch.from_numpy(test_t["data"]).float()

        test_img = np.load(i_file)
        self.test_data_img = torch.from_numpy(test_img["data"]).squeeze().float()

        self.test_labels = torch.from_numpy(test_t["label"]).long()

    def __len__(self):

        return self.test_data_t.shape[0]

    def __getitem__(self, item):

        return self.test_data_t[item], self.test_data_img[item], self.test_labels[item]


