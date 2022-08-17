
import copy
import pickle as pickle
from random import sample
import torchvision
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import datetime
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from transformers import BertModel, BertTokenizer
import clip
from transformers import pipeline
from googletrans import Translator
# from logger import Logger
import models_mae
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import scipy.io as sio
from mdfend.layers import MaskAttention, cnn_extractor
from timm.models.vision_transformer import Block
class UAMFD_Net(nn.Module):
    def __init__(self, dataset='weibo',hidden_dim=256):
        # USING WWW DATA LOADER
        # NOTE: NOW WE ONLY SUPPORT BASE MODEL!
        model_size = 'base'
        self.model_size = model_size
        self.dataset = dataset
        self.num_expert = 5
        self.unified_dim, self.text_dim = 768, 200
        self.hidden_size = hidden_dim
        super(UAMFD_Net, self).__init__()
        # IMAGE: MAE

        # INPUT IS (BATCH, 512)
        self.image_model = nn.Sequential(nn.Linear(512, self.unified_dim),
                                  nn.BatchNorm1d(self.unified_dim),
                                  nn.GELU(),
                                  nn.Linear(self.unified_dim, self.unified_dim),
                                  )

        self.image_model.cuda()

        # TEXT: BERT OR PRETRAINED FROM WWW

        self.convert_mlp = nn.Sequential(
                nn.Linear(self.text_dim, self.unified_dim),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(self.unified_dim, self.unified_dim),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(self.unified_dim, self.unified_dim),
            )
        text_transblocks, depth = [], 4
        for j in range(depth):
            text_transblocks.append(Block(dim=self.unified_dim, num_heads=16))
        self.text_model = nn.ModuleList(text_transblocks)
        self.text_attention = MaskAttention(self.unified_dim)

        # IMAGE: RESNET-50

        # INPUT IS (BATCH, 512)
        self.netG = nn.Sequential(nn.Linear(512, self.unified_dim),
                                    nn.BatchNorm1d(self.unified_dim),
                                    nn.GELU(),
                                    nn.Linear(self.unified_dim, self.unified_dim),
                                    )
        self.netG = self.netG.cuda()
        self.image_attention = MaskAttention(self.unified_dim)
        # GATE, EXPERTS
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64} # 64*5 note there are 5 kernels and 5 experts!
        image_expert, text_expert, mm_expert = [], [], []
        depth = 1
        for i in range(self.num_expert-1):
            for j in range(depth):
                 image_expert.append(nn.Sequential(
                                        nn.Linear(self.unified_dim, self.unified_dim),
                                        nn.BatchNorm1d(self.unified_dim),
                                        nn.GELU(),
                                        nn.Dropout(0.25),
                                        nn.Linear(self.unified_dim, self.unified_dim))
                    )

        for i in range(self.num_expert):
            for j in range(depth):
                text_expert.append(Block(dim=self.unified_dim, num_heads=16))
                mm_expert.append(Block(dim=self.unified_dim, num_heads=16))
                # text_expert.append(cnn_extractor(feature_kernel, self.unified_dim))
                # mm_expert.append(cnn_extractor(feature_kernel, self.unified_dim + self.unified_dim))


        self.image_experts = nn.ModuleList(image_expert)
        self.text_experts = nn.ModuleList(text_expert)
        self.mm_experts = nn.ModuleList(mm_expert)
        # self.out_unified_dim = 320
        self.image_gate_mae = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                        nn.BatchNorm1d(self.unified_dim),
                                        nn.GELU(),
                                        nn.Dropout(0.25),
                                        nn.Linear(self.unified_dim, self.num_expert-1),
                                        )
        self.image_gate_vgg = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                        nn.BatchNorm1d(self.unified_dim),
                                        nn.GELU(),
                                        nn.Dropout(0.25),
                                        nn.Linear(self.unified_dim, 1),
                                        )
        self.soft_max = nn.Softmax(dim=1)
        self.text_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                      nn.BatchNorm1d(self.unified_dim),
                                      nn.GELU(),
                                      nn.Dropout(0.25),
                                      nn.Linear(self.unified_dim, self.num_expert),
                                      nn.Softmax(dim=1))
        self.mm_gate = nn.Sequential(nn.Linear(self.unified_dim+self.unified_dim, self.unified_dim),
                                     nn.BatchNorm1d(self.unified_dim),
                                     nn.GELU(),
                                     nn.Dropout(0.25),
                                     nn.Linear(self.unified_dim, self.num_expert),
                                     nn.Softmax(dim=1))
        # FUSION SCORE GATE
        self.mm_score_classifier = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                     nn.BatchNorm1d(self.unified_dim),
                                     nn.GELU(),
                                     nn.Dropout(0.25),
                                     nn.Linear(self.unified_dim, 2)
                                           )

        # CLASSIFICATION HEAD
        self.text_alone_attn = nn.Sequential(
            nn.Linear(self.unified_dim,self.unified_dim),
            nn.BatchNorm1d(self.unified_dim),
            nn.GELU(),
            # nn.Dropout(0.25),
        )
        self.image_alone_attn = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            # nn.Dropout(0.25),
            nn.BatchNorm1d(self.unified_dim),
            nn.GELU(),
        )
        self.mix_attn = nn.Sequential(
            nn.Linear(3*self.unified_dim, self.unified_dim ),
            # nn.Dropout(0.25),
            nn.BatchNorm1d(self.unified_dim ),
            nn.GELU(),
        )
        self.image_alone_classifier = nn.Sequential(
            nn.Linear(self.unified_dim,2)
        )
        self.text_alone_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 2)
        )
        self.mix_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            # nn.Dropout(0.25),
            nn.BatchNorm1d(self.unified_dim),
            nn.GELU(),
            nn.Linear(self.unified_dim, 2),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, image, imageclip=None, textclip=None):

        # BASE FEATURE AND ATTENTION

        # IMAGE:  OUTPUT IS (BATCH, 512)
        image_feature = self.image_model(image)



        # TEXT:  INPUT IS (BATCH, WORDLEN, 200)
        converted_input_ids = input_ids.reshape(input_ids.shape[0]*input_ids.shape[1],-1)
        converted_input_ids = self.convert_mlp(converted_input_ids)
        converted_input_ids = converted_input_ids.reshape(input_ids.shape[0],input_ids.shape[1],-1)
        # print("converted size {}".format(converted_input_ids.shape))
        text_feature = converted_input_ids
        for i in range(len(self.text_model)):
            text_feature = self.text_model[i](text_feature)

        # print("text_feature size {}".format(text_feature.shape)) # 64,170,768
        # print("image_feature size {}".format(image_feature.shape)) # 64,197,1024
        text_atn_feature, _ = self.text_attention(text_feature, attention_mask)

        # IMAGE ATTENTION NOT NEEDED BY WWW LOADER
        image_atn_feature = image_feature

        vgg_feature = self.netG(image)  # 64, 768
        # print("text_atn_feature size {}".format(text_atn_feature.shape)) # 64, 768
        # print("image_atn_feature size {}".format(image_atn_feature.shape))
        # GATE
        gate_image_feature_mae = self.image_gate_mae(image_atn_feature)
        gate_image_feature_vgg = self.image_gate_vgg(vgg_feature)
        gate_image_feature = self.soft_max(torch.cat((gate_image_feature_mae,gate_image_feature_vgg),dim=1))
        gate_text_feature = self.text_gate(text_atn_feature) # 64 320
        gate_mm_feature = self.mm_gate(torch.cat((image_atn_feature, text_atn_feature),dim=1))
        # IMAGE EXPERTS
        # NOTE: IMAGE/TEXT/MM EXPERTS WILL BE MLPS IF WE USE WWW LOADER
        shared_image_feature = 0
        for i in range(self.num_expert-1):
            tmp_image_feature = self.image_experts[i](image_feature)
            shared_image_feature += (tmp_image_feature * gate_image_feature[:, i].unsqueeze(1))
        # print(vgg_feature.shape)
        # print(shared_image_feature.shape)
        # print(gate_image_feature.shape)
        shared_image_feature += (vgg_feature * gate_image_feature[:, -1].unsqueeze(1))

        # TEXT AND MM EXPERTS

        shared_mm_feature, shared_text_feature = 0, 0
        for i in range(self.num_expert):
            tmp_text_feature = self.text_experts[i](text_feature)[:,0] # text_feature: 64, 170, 768
            shared_text_feature += (tmp_text_feature * gate_text_feature[:, i].unsqueeze(1))


        shared_mm_feature = (shared_text_feature+shared_image_feature)/2

        # GATED SCORE
        # gated_score = self.mm_score_classifier(shared_mm_feature)


        # print("shared_image_feature {}".format(shared_image_feature.shape))
        # print("shared_text_feature {}".format(shared_text_feature.shape))
        ### text-only branch
        text_alone_output = self.text_alone_classifier(self.text_alone_attn(shared_text_feature))
        ### image-only branch
        image_alone_output = self.image_alone_classifier(self.image_alone_attn(shared_image_feature))

        ### mixed branch
        concat_feature = torch.cat((shared_image_feature, shared_text_feature, shared_mm_feature), dim=1)
        mix_output = self.mix_classifier(self.mix_attn(concat_feature))

        return text_alone_output, image_alone_output, mix_output
