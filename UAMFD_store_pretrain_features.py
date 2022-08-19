import numpy as np
import argparse
import time, os
from sklearn import metrics
import copy
import pickle as pickle
from random import sample
import torchvision
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import pytorch_warmup as warmup
from loss.focal_loss import focal_loss

GT_size = 224
word_token_length = 197 # identical to size of MAE
image_token_length = 197
token_chinese = BertTokenizer.from_pretrained('bert-base-chinese')
token_uncased = BertTokenizer.from_pretrained('bert-base-uncased')

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def collate_fn_english(data):
    item = data[0]
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    labels = [i[0][2] for i in data]
    category = [0 for i in data]
    GT_path = [i[1] for i in data]
    token_data = token_uncased.batch_encode_plus(batch_text_or_text_pairs=sents,
                                                 truncation=True,
                                                 padding='max_length',
                                                 max_length=word_token_length,
                                                 return_tensors='pt',
                                                 return_length=True)

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = token_data['input_ids']
    attention_mask = token_data['attention_mask']
    token_type_ids = token_data['token_type_ids']
    image = torch.stack(image)
    labels = torch.LongTensor(labels)
    category = torch.LongTensor(category)

    if len(item) <= 3:
        return (input_ids, attention_mask, token_type_ids), (image, labels, category, sents), GT_path
    else:
        sents1 = [i[2][0] for i in data]
        image1 = [i[2][1] for i in data]
        labels1 = [i[2][2] for i in data]
        token_data1 = token_chinese.batch_encode_plus(batch_text_or_text_pairs=sents1,
                                                      truncation=True,
                                                      padding='max_length',
                                                      max_length=word_token_length,
                                                      return_tensors='pt',
                                                      return_length=True)

        input_ids1 = token_data1['input_ids']
        attention_mask1 = token_data1['attention_mask']
        token_type_ids1 = token_data1['token_type_ids']
        image1 = torch.stack(image1)
        labels1 = torch.LongTensor(labels1)

        return (input_ids, attention_mask, token_type_ids), (image, labels, category, sents), GT_path, \
               (input_ids1, attention_mask1, token_type_ids1), (image1, labels1, sents1)

def collate_fn_chinese(data):
    """ In Weibo dataset
        if not self.with_ambiguity:
            return (content, img_GT, label, 0), (GT_path)
        else:
            return (content, img_GT, label, 0), (GT_path), (content_ambiguity, img_ambiguity, label_ambiguity)
    """
    item = data[0]
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    labels = [i[0][2] for i in data]
    category = [0 for i in data]
    GT_path = [i[1] for i in data]
    token_data = token_chinese.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=word_token_length,
                                   return_tensors='pt',
                                   return_length=True)

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = token_data['input_ids']
    attention_mask = token_data['attention_mask']
    token_type_ids = token_data['token_type_ids']
    image = torch.stack(image)
    labels = torch.LongTensor(labels)
    category = torch.LongTensor(category)

    if len(item) <= 3:
        return (input_ids, attention_mask, token_type_ids), (image, labels, category, sents), GT_path
    else:
        sents1 = [i[2][0] for i in data]
        image1 = [i[2][1] for i in data]
        labels1 = [i[2][2] for i in data]
        token_data1 = token_chinese.batch_encode_plus(batch_text_or_text_pairs=sents1,
                                                      truncation=True,
                                                      padding='max_length',
                                                      max_length=word_token_length,
                                                      return_tensors='pt',
                                                      return_length=True)

        input_ids1 = token_data1['input_ids']
        attention_mask1 = token_data1['attention_mask']
        token_type_ids1 = token_data1['token_type_ids']
        image1 = torch.stack(image1)
        labels1 = torch.LongTensor(labels1)

        return (input_ids, attention_mask, token_type_ids), (image, labels, category, sents), GT_path, \
               (input_ids1, attention_mask1, token_type_ids1), (image1, labels1, sents1)

from torch.utils.tensorboard import SummaryWriter
from utils import Progbar, create_dir, stitch_images, imsave
stateful_metrics = ['L-RealTime','lr','APEXGT','empty','exclusion','FW1', 'QF','QFGT','QFR','BK1', 'FW', 'BK','FW1', 'BK1', 'LC', 'Kind',
                                'FAB1','BAB1','A', 'AGT','1','2','3','4','0','gt','pred','RATE','SSBK']

#网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def main(args):
    print(args)

    # world_size, rank = init_dist()

    use_scalar = False
    if use_scalar:
        writer = SummaryWriter(f'runs/mae-main')
    seed = 25
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## Slower but more reproducible
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    ## Faster but less reproducible
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print("Using amp (Tempt)")
    scaler = torch.cuda.amp.GradScaler()

    print('loading data')
    ############### SETTINGS ###################
    ## DATASETS AVALIABLE: WWW, weibo, gossip, politi, Twitter, Mix
    setting = {}
    setting['checkpoint_path'] = args.checkpoint #''
    # setting['checkpoint_path'] = '/home/groupshare/CIKM_ying_output/weibo/35_68_91.pkl'
    # setting['checkpoint_path'] = '/home/groupshare/CIKM_ying_output/gossip/1_612_87.pkl'
    print('loading checkpoint from {}'.format(setting['checkpoint_path']))
    setting['train_dataname'] = args.train_dataset
    setting['val_dataname'] = args.test_dataset
    setting['is_filter'] = False
    setting['duplicate_fake_times'] = 1
    setting['is_use_unimodal'] = True
    setting['with_ambiguity'] = False
    # CURRENTLY ONLY SUPPORT GOSSIP Weibo
    LIST_ALLOW_AMBIGUITY = ['gossip','weibo']
    setting['with_ambiguity'] = setting['with_ambiguity'] and setting['train_dataname'] in LIST_ALLOW_AMBIGUITY
    setting['data_augment'] = False
    setting['is_use_bce'] = True
    setting['use_soft_label'] = False
    setting['is_sample_positive'] = 0.4 #if setting['train_dataname'] != 'gossip' else 0.3
    LOW_BATCH_SIZE_AND_LR = ['Twitter','politi']
    custom_batch_size = args.batch_size
        #8 if setting['train_dataname'] in LOW_BATCH_SIZE_AND_LR else 32
    custom_lr = 1e-4 if setting['train_dataname'] in LOW_BATCH_SIZE_AND_LR else 5e-5
    custom_num_epochs = args.epochs
        #50 if setting['train_dataname'] in LOW_BATCH_SIZE_AND_LR else 100
    #############################################
    print("Filter the dataset? {}".format(setting['is_filter']))
    is_use_WWW_loader = setting['train_dataname']=='WWW'
    train_dataset, validate_dataset, train_loader, validate_loader = None,None,None,None
    shuffle, num_workers = True, 4
    train_sampler = None


    ########## train dataset ####################
    if setting['train_dataname']=='weibo':
        print("Using weibo as training")
        # Note: bert-base-chinese is within MixSet_dataset
        from data.weibo_dataset import weibo_dataset
        train_dataset = weibo_dataset(is_train=True, image_size=GT_size,
                                      with_ambiguity=setting['with_ambiguity']
                                      )

        train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=shuffle,
                                  collate_fn=collate_fn_chinese,
                                  num_workers=num_workers, sampler=train_sampler, drop_last=True,
                                  pin_memory=True)

    elif setting['train_dataname']=="WWW":
        from data.FeatureDataSet import FeatureDataSet
        NUM_WORKER = 4
        dataset_dir = '/home/groupshare/WWW_rumor_detection'

        train_dataset = FeatureDataSet(
            "{}/train_text+label.npz".format(dataset_dir),
            "{}/train_image+label.npz".format(dataset_dir), )
        train_loader = DataLoader(
            train_dataset,
            batch_size=custom_batch_size,
            num_workers=NUM_WORKER,
            shuffle=True)

    elif setting['train_dataname']=='Twitter':
        print("Using Twitter as training")
        # Note: bert-base-chinese is within MixSet_dataset
        from data.Twitter_dataset import Twitter_dataset
        train_dataset = Twitter_dataset(is_train=True, image_size=GT_size)
        train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=shuffle, collate_fn=collate_fn_english,
                                  num_workers=num_workers, sampler=train_sampler, drop_last=True,
                                  pin_memory=True)


    elif setting['train_dataname']=='Mix':
        print("Using MixSet as training")
        # Note: bert-base-chinese is within MixSet_dataset
        from data.MixSet_dataset import MixSet_dataset
        train_dataset = MixSet_dataset(is_train=True, image_size=GT_size)
        train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=shuffle,collate_fn=collate_fn_chinese,
                                           num_workers=num_workers, sampler=train_sampler, drop_last=True,
                                           pin_memory=True)


    else:
        print("Using FakeNewsNet as training")
        # Note: bert-base-chinese is within MixSet_dataset
        from data.FakeNet_dataset import FakeNet_dataset
        train_dataset = FakeNet_dataset(is_filter=setting['is_filter'],
                                        is_train=True,
                                        is_use_unimodal=setting['is_use_unimodal'],
                                        dataset=setting['train_dataname'],
                                        image_size=GT_size,
                                        data_augment = setting['data_augment'],
                                        with_ambiguity=setting['with_ambiguity'],
                                        use_soft_label=setting['use_soft_label'],
                                        is_sample_positive=setting['is_sample_positive'],
                                        )
        train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=True, collate_fn=collate_fn_english,
                                  num_workers=4, sampler=None, drop_last=True,
                                  pin_memory=True)
    ########## validate dataset ####################
    if setting['val_dataname']=='weibo':
        print("Using weibo as inference")
        from data.weibo_dataset import weibo_dataset
        validate_dataset = weibo_dataset(is_train=False, image_size=GT_size)
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                     collate_fn=collate_fn_chinese,
                                     num_workers=4, sampler=None, drop_last=True,
                                     pin_memory=True)

    elif setting['val_dataname']=="WWW":
        from data.FeatureDataSet import FeatureDataSet
        NUM_WORKER = 4
        dataset_dir = './'
        validate_dataset = FeatureDataSet(
            "{}/test_text+label.npz".format(dataset_dir),
            "{}/test_image+label.npz".format(dataset_dir), )
        validate_loader = DataLoader(
            validate_dataset, batch_size=custom_batch_size, num_workers=NUM_WORKER)

    elif setting['val_dataname']=='Twitter':
        from data.Twitter_dataset import Twitter_dataset
        print("Using Twitter as inference")
        validate_dataset = Twitter_dataset(is_train=False, image_size=GT_size)
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                     collate_fn=collate_fn_english,
                                     num_workers=4, sampler=None, drop_last=True,
                                     pin_memory=True)

    elif setting['val_dataname']=='Mix':
        from data.MixSet_dataset import MixSet_dataset
        print("using Mix as inference")
        validate_dataset = MixSet_dataset(is_train=False, image_size=GT_size)
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,collate_fn=collate_fn_chinese,
                                           num_workers=4, sampler=None, drop_last=True,
                                           pin_memory=True)
    else:
        from data.FakeNet_dataset import FakeNet_dataset
        print("using FakeNet as inference")
        validate_dataset = FakeNet_dataset(is_filter=False, is_train=False,
                                           dataset=setting['val_dataname'],
                                           is_use_unimodal=setting['is_use_unimodal'],
                                           image_size=GT_size)
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                     collate_fn=collate_fn_english,
                                     num_workers=4, sampler=None, drop_last=True,
                                     pin_memory=True)


    ############## MODEL SELECTION #############################
    print('building model')

    from models.UAMFDv2_Net import UAMFD_Net
    ## V2 is always used for innovation
    # from models.UAMFDv2_Net import UAMFD_Net
    model = UAMFD_Net(dataset=setting['train_dataname'],
                      text_token_len=word_token_length,
                      image_token_len=image_token_length,
                      is_use_bce=setting['is_use_bce'],
                      batch_size=custom_batch_size,
                      )

    if len(setting['checkpoint_path'])!=0:
        print("loading checkpoint: {}".format(setting['checkpoint_path']))
        load_model(model, setting['checkpoint_path'])
    model = model.cuda()
    model.eval()

    print(get_parameter_number(model))
    ############################################################
    ##################### Loss and Optimizer ###################


    loaders = [train_loader, validate_loader]
    for idx, loader in enumerate(loaders):
        pickle_dict = {}
        for epoch in range(custom_num_epochs):
            total = len(train_dataset if idx==0 else validate_dataset)
            progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            for i, items in enumerate(loader):
                with torch.no_grad():
                    logs = []
                    texts, others, GT_path = items
                    input_ids, attention_mask, token_type_ids = texts
                    image, labels, category, sents = others
                    input_ids, attention_mask, token_type_ids, image, labels, category = \
                        to_var(input_ids), to_var(attention_mask), to_var(token_type_ids), \
                        to_var(image), to_var(labels), to_var(category)

                    if GT_path[0] not in pickle_dict or sents[0] not in pickle_dict:
                        bert_features, mae_features = model.get_pretrain_features(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              token_type_ids=token_type_ids,
                                              image=image,
                                              no_ambiguity=not setting['with_ambiguity'],
                                              category=category,
                                              calc_ambiguity=False,
                                          )
                        # bert_features = bert_features.cpu().numpy()
                        # mae_features = mae_features.cpu().numpy()
                        if GT_path[0] not in pickle_dict:
                            pickle_dict[GT_path[0]] = mae_features.squeeze(0).cpu().numpy()
                            print(f"Added {GT_path[0]}")
                        if sents[0] not in pickle_dict:
                            pickle_dict[sents[0]] = bert_features.squeeze(0).cpu().numpy()
                            print(f"Added {sents[0]}")
                    progbar.add(len(image), values=logs)

        train_or_test = "train" if idx==0 else "val"
        filename = f"./{train_or_test}_{setting['train_dataname']}"
        with open(f"{filename}.pickle",'wb') as f:
            pickle.dump(pickle_dict, f)
        print(f"Saved {filename}")
                    
                    
                    
                    

from collections import OrderedDict
def load_model(model, load_path, strict=False):
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=strict)




def load_data(args, dataset):
    if dataset=='weibo':
        import process_data_weibo as process_data
        train, validate = process_data.get_data(args.text_only)
    else: # "Twitter"
        import process_data_Twitter as process_data
        train, validate = process_data.get_data(args.text_only)

    # f = open('/home/groupshare/ITCN/train.pckl','rb')
    # train = pickle.load(f)
    # f.close()
    #
    # f = open('/home/groupshare/ITCN/validate.pckl','rb')
    # validate = pickle.load(f)
    # f.close()
    #
    # f = open('test.pckl','rb')
    # test = pickle.load(f)
    # f.close()

    # print(train[4][0])
    args.vocab_size = 25
    args.sequence_len = 25

    print("sequence length " + str(args.sequence_length))
    print("Train Data Size is " + str(len(train['post_text'])))
    print("Finished loading data ")
    # return train,validate, test

    return train, validate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-training_file', type=str, default='', help='')
    parser.add_argument('-validation_file', type=str, default='', help='')
    parser.add_argument('-testing_file', type=str, default='', help='')
    parser.add_argument('-output_file', type=str, default='/home/groupshare/CIKM_ying_output/', help='')
    # parser.add_argument('-dataset', type=str, default='weibo', help='')
    parser.add_argument('-train_dataset', type=str, default='Twitter', help='')
    parser.add_argument('-test_dataset', type=str, default='Twitter', help='')
    parser.add_argument('-checkpoint', type=str, default='', help='')
    parser.add_argument('-static', type=bool, default=True, help='')
    parser.add_argument('-sequence_length', type=int, default=25, help='')
    parser.add_argument('-finetune', type=int, default=0, help='')
    parser.add_argument('-class_num', type=int, default=2, help='')
    parser.add_argument('-batch_size', type=int, default=16, help='')
    parser.add_argument('-epochs', type=int, default=100, help='')
    parser.add_argument('-hidden_dim', type=int, default=512, help='')
    parser.add_argument('-embed_dim', type=int, default=32, help='')
    parser.add_argument('-vocab_size', type=int, default=25, help='')
    parser.add_argument('-lambd', type=int, default=1, help='')
    parser.add_argument('-text_only', type=bool, default=False, help='')
    args = parser.parse_args()

    main(args)


