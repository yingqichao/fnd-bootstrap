
import cv2

import torch
import torch.utils.data as data
import data.util as util

import torchvision.transforms.functional as F

from PIL import Image
import os
import openpyxl
import pandas as pd
import numpy as np
import paramiko
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import clip


class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.transform = transforms.Compose([
            transforms.Resize(GT_size),
            transforms.CenterCrop(GT_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clipmodel, self.preprocess = clip.load('ViT-B/32', self.device)
        self.text = list(dataset['post_text'])
        self.image = list(dataset['image'])
        self.trantext = list(dataset['post_text']) if 'tran_texts' not in dataset else list(dataset['tran_texts'])
        self.label = torch.from_numpy(np.array(dataset['label']))
        # print('TEXT: %d, Image: %d, labe: %d'
        #       % (len(self.text), len(self.image), len(self.label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):


        return (self.text[idx], self.transform(self.image[idx]), self.preprocess(self.image[idx]), self.trantext[idx]), \
               self.label[idx]