import numpy as np
import argparse
import time, os
from sklearn import metrics
import copy
import pickle as pickle
from random import sample
import torchvision
from sklearn.model_selection import train_test_split
from models.Mrmu_1010 import SimilarityPart, MultiModal
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
word_token_length = 197  # identical to size of MAE
image_token_length = 197
token_chinese = BertTokenizer.from_pretrained('bert-base-chinese')
token_uncased = BertTokenizer.from_pretrained('bert-base-uncased')


# clipmodel, preprocess = clip.load('ViT-B/32', device)
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# translator = Translator(service_urls=[
#     'translate.google.cn'
# ])


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


# def collate_fn_weibo(data):
#     sents = [i[0][0] for i in data]
#     image = [i[0][1] for i in data]
#     imageclip = [i[0][2] for i in data]
#     textclip = [i[0][3] for i in data]
#     labels = [i[1] for i in data]
#     data = token_chinese.batch_encode_plus(batch_text_or_text_pairs=sents,
#                                    truncation=True,
#                                    padding='max_length',
#                                    max_length=word_token_length,
#                                    return_tensors='pt',
#                                    return_length=True)
#
#     textclip = clip.tokenize(textclip, truncate=True)
#     # input_ids:编码之后的数字
#     # attention_mask:是补零的位置是0,其他位置是1
#     input_ids = data['input_ids']
#     attention_mask = data['attention_mask']
#     token_type_ids = data['token_type_ids']
#     image = torch.stack(image)
#     imageclip = torch.stack(imageclip)
#     labels = torch.LongTensor(labels)
#
#     # print(data['length'], data['length'].max())
#
#     return input_ids, attention_mask, token_type_ids, image, imageclip, textclip, labels

def collate_fn_english(data):
    item = data[0]
    if len(item) == 2:
        category = [0 for i in data]
    else:
        category = [i[2] for i in data]
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    labels = [i[1] for i in data]

    token_data = token_uncased.batch_encode_plus(batch_text_or_text_pairs=sents,
                                                 truncation=True,
                                                 padding='max_length',
                                                 max_length=word_token_length,
                                                 return_tensors='pt',
                                                 return_length=True)

    input_ids = token_data['input_ids']
    attention_mask = token_data['attention_mask']
    token_type_ids = token_data['token_type_ids']
    image = torch.stack(image)
    labels = torch.LongTensor(labels)
    category = torch.LongTensor(category)
    if len(item) <= 3:
        return (input_ids, attention_mask, token_type_ids), image, labels, category
    else:
        sents1 = [i[3][0] for i in data]
        image1 = [i[3][1] for i in data]
        labels1 = [i[4] for i in data]
        token_data1 = token_uncased.batch_encode_plus(batch_text_or_text_pairs=sents1,
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

        return (input_ids, attention_mask, token_type_ids), image, labels, category, \
               (input_ids1, attention_mask1, token_type_ids1), image1, labels1


def collate_fn_chinese(data):
    item = data[0]
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    labels = [i[1] for i in data]
    if len(item) == 2:
        category = [0 for i in data]
    else:
        category = [i[2] for i in data]
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
        return (input_ids, attention_mask, token_type_ids), image, labels, category
    else:
        sents1 = [i[3][0] for i in data]
        image1 = [i[3][1] for i in data]
        labels1 = [i[4] for i in data]
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

        return (input_ids, attention_mask, token_type_ids), image, labels, category, \
               (input_ids1, attention_mask1, token_type_ids1), image1, labels1


from utils import Progbar, create_dir, stitch_images, imsave

stateful_metrics = ['L-RealTime', 'lr', 'APEXGT', 'empty', 'exclusion', 'FW1', 'QF', 'QFGT', 'QFR', 'BK1', 'FW', 'BK',
                    'FW1', 'BK1', 'LC', 'Kind',
                    'FAB1', 'BAB1', 'A', 'AGT', '1', '2', '3', '4', '0', 'gt', 'pred', 'RATE', 'SSBK']

def prepare_task2_data(text, image, label):
    nr_index = [i for i, la in enumerate(label) if la == 1]
    text_nr = text[nr_index]
    image_nr = image[nr_index]
    fixed_text = copy.deepcopy(text_nr)
    matched_image = copy.deepcopy(image_nr)
    unmatched_image = copy.deepcopy(image_nr).roll(shifts=3, dims=0)
    return fixed_text, matched_image, unmatched_image

def main(args):
    print('loading data')
    ############### SETTINGS ###################
    ## DATASETS AVALIABLE: WWW, weibo, gossip, politi, Twitter, Mix
    english_lists = ['gossip','Twitter','politi']
    setting = {}
    setting['checkpoint_path'] = ''
    setting['train_dataname'] = 'Twitter'  # args.dataset
    setting['val_dataname'] = 'Twitter'
    setting['is_filter'] = False
    setting['is_use_unimodal'] = True
    setting['with_ambiguity'] = False
    # CURRENTLY ONLY SUPPORT GOSSIP Weibo
    LIST_ALLOW_AMBIGUITY = ['gossip', 'weibo']
    setting['with_ambiguity'] = setting['with_ambiguity'] and setting['train_dataname'] in LIST_ALLOW_AMBIGUITY
    setting['data_augment'] = False
    setting['is_use_bce'] = True
    setting['use_soft_label'] = False
    setting['is_sample_positive'] = 0.3
    LOW_BATCH_SIZE_AND_LR = ['Twitter', 'politi']
    custom_batch_size = 128 if setting['train_dataname'] in LOW_BATCH_SIZE_AND_LR else 32
    custom_lr = 1e-3 if setting['train_dataname'] in LOW_BATCH_SIZE_AND_LR else 1e-3
    custom_num_epochs = 150 if setting['train_dataname'] in LOW_BATCH_SIZE_AND_LR else 100
    #############################################
    print("Filter the dataset? {}".format(setting['is_filter']))
    is_use_WWW_loader = setting['train_dataname'] == 'WWW'
    train_dataset, validate_dataset, train_loader, validate_loader = None, None, None, None
    ########## train dataset ####################
    if setting['train_dataname'] == 'weibo':
        print("Using weibo as training")
        # Note: bert-base-chinese is within MixSet_dataset
        from data.weibo_dataset import weibo_dataset
        train_dataset = weibo_dataset(is_train=True, image_size=GT_size,
                                      with_ambiguity=setting['with_ambiguity']
                                      )
        train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=True,
                                  collate_fn=collate_fn_chinese,
                                  num_workers=4, sampler=None, drop_last=True,
                                  pin_memory=False)
        # # train, validation, test = load_data(args)
        # train, validation = load_data(args,'weibo')
        # train_dataset = Rumor_Data(train)
        # validate_dataset = Rumor_Data(validation)
        # # test_dataset = Rumor_Data(test)
        # train_loader = DataLoader(dataset=train_dataset,
        #                           batch_size=custom_batch_size,
        #                           collate_fn=collate_fn_weibo,
        #                           shuffle=True,drop_last=True)
        # validate_loader = DataLoader(dataset=validate_dataset,
        #                              batch_size=custom_batch_size,
        #                              collate_fn=collate_fn_weibo,
        #                              shuffle=False,drop_last=True)
    elif setting['train_dataname'] == "WWW":
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

    elif setting['train_dataname'] == 'Twitter':
        print("Using Twitter as training")
        # Note: bert-base-chinese is within MixSet_dataset
        from data.Twitter_dataset import Twitter_dataset
        train_dataset = Twitter_dataset(is_train=True, image_size=GT_size, remove_duplicate=False)
        train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=True,
                                  collate_fn=collate_fn_english,
                                  num_workers=4, sampler=None, drop_last=True,
                                  pin_memory=False)

        # OLD LOADING
        # train, validation = load_data(args,'Twitter')
        # train_dataset = Rumor_Data(train)
        # validate_dataset = Rumor_Data(validation)
        # # test_dataset = Rumor_Data(test)
        # train_loader = DataLoader(dataset=train_dataset,
        #                           batch_size=custom_batch_size,
        #                           collate_fn=collate_fn_english,
        #                           shuffle=True,drop_last=True)
        # validate_loader = DataLoader(dataset=validate_dataset,
        #                              batch_size=custom_batch_size,
        #                              collate_fn=collate_fn_english,
        #                              shuffle=False,drop_last=True)
    elif setting['train_dataname'] == 'Mix':
        print("Using MixSet as training")
        # Note: bert-base-chinese is within MixSet_dataset
        from data.MixSet_dataset import MixSet_dataset
        train_dataset = MixSet_dataset(is_train=True, image_size=GT_size)
        train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=True,
                                  collate_fn=collate_fn_chinese,
                                  num_workers=4, sampler=None, drop_last=True,
                                  pin_memory=False)

        # train_puretext_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=True,collate_fn=collate_fn_chinese,
        #                                    num_workers=4, sampler=None, drop_last=True,
        #                                    pin_memory=False)
        #
        # train_pureimage_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=True,
        #                                    collate_fn=collate_fn_chinese,
        #                                    num_workers=4, sampler=None, drop_last=True,
        #                                    pin_memory=False)
    else:
        print("Using FakeNewsNet as training")
        # Note: bert-base-chinese is within MixSet_dataset
        from data.FakeNet_dataset import FakeNet_dataset
        train_dataset = FakeNet_dataset(is_filter=setting['is_filter'],
                                        is_train=True,
                                        is_use_unimodal=setting['is_use_unimodal'],
                                        dataset=setting['train_dataname'],
                                        image_size=GT_size,
                                        data_augment=setting['data_augment'],
                                        with_ambiguity=setting['with_ambiguity'],
                                        use_soft_label=setting['use_soft_label'],
                                        is_sample_positive=setting['is_sample_positive'],
                                        )
        train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=True,
                                  collate_fn=collate_fn_english,
                                  num_workers=4, sampler=None, drop_last=True,
                                  pin_memory=False)
    ########## validate dataset ####################
    if setting['val_dataname'] == 'weibo':
        print("Using weibo as inference")
        from data.weibo_dataset import weibo_dataset
        validate_dataset = weibo_dataset(is_train=False, image_size=GT_size)
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                     collate_fn=collate_fn_chinese,
                                     num_workers=4, sampler=None, drop_last=False,
                                     pin_memory=False)
        # _, validation = load_data(args, 'weibo')
        # validate_dataset = Rumor_Data(validation)
        # validate_loader = DataLoader(dataset=validate_dataset,
        #                              batch_size=custom_batch_size,
        #                              collate_fn=collate_fn_weibo,
        #                              shuffle=False, drop_last=True)
    elif setting['val_dataname'] == "WWW":
        from data.FeatureDataSet import FeatureDataSet
        NUM_WORKER = 4
        dataset_dir = './'
        validate_dataset = FeatureDataSet(
            "{}/test_text+label.npz".format(dataset_dir),
            "{}/test_image+label.npz".format(dataset_dir), )
        validate_loader = DataLoader(
            validate_dataset, batch_size=custom_batch_size, num_workers=NUM_WORKER)

    elif setting['val_dataname'] == 'Twitter':
        from data.Twitter_dataset import Twitter_dataset
        print("Using Twitter as inference")
        validate_dataset = Twitter_dataset(is_train=False, image_size=GT_size)
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                     collate_fn=collate_fn_english,
                                     num_workers=4, sampler=None, drop_last=False,
                                     pin_memory=False)

    elif setting['val_dataname'] == 'Mix':
        from data.MixSet_dataset import MixSet_dataset
        print("using Mix as inference")
        validate_dataset = MixSet_dataset(is_train=False, image_size=GT_size)
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                     collate_fn=collate_fn_chinese,
                                     num_workers=4, sampler=None, drop_last=False,
                                     pin_memory=False)
    else:
        from data.FakeNet_dataset import FakeNet_dataset
        print("using FakeNet as inference")
        validate_dataset = FakeNet_dataset(is_filter=False, is_train=False,
                                           dataset=setting['val_dataname'],
                                           is_use_unimodal=setting['is_use_unimodal'],
                                           image_size=GT_size)
        validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
                                     collate_fn=collate_fn_english,
                                     num_workers=4, sampler=None, drop_last=False,
                                     pin_memory=False)
        # puretext_iter = train_puretext_loader.__iter__()
        # pureimage_iter = train_pureimage_loader.__iter__()

    ############## MODEL SELECTION #############################
    # print('building model')
    # if is_use_WWW_loader:
    #     from models.UAMFDforWWW_Net import UAMFD_Net
    #     model = UAMFD_Net(dataset=setting['train_dataname'], is_use_bce=setting['is_use_bce'])
    # else:
    #     from models.UAMFD_Net import UAMFD_Net
    #     model = UAMFD_Net(dataset=setting['train_dataname'],
    #                       text_token_len=word_token_length,
    #                       image_token_len=image_token_length,
    #                       is_use_bce=setting['is_use_bce']
    #                       )
    #
    # if len(setting['checkpoint_path']) != 0:
    #     print("loading checkpoint: {}".format(setting['checkpoint_path']))
    #     load_model(model, setting['checkpoint_path'])
    # model = model.cuda()
    # model.train()
    ############################################################
    ##################### Loss and Optimizer ###################
    loss_cross_entropy = nn.CrossEntropyLoss().cuda()
    loss_focal = focal_loss(alpha=0.25, gamma=2, num_classes=2).cuda()
    loss_bce = nn.BCELoss().cuda()
    criterion = loss_bce if setting['is_use_bce'] else loss_focal
    l1_loss = nn.L1Loss().cuda()
    print("Using Focal Loss.")
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, list(model.parameters())),
    #                               lr=custom_lr, betas=(0.9, 0.999), weight_decay=0.01)
    # scheduler = ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=3)
    # scheduler = MultiStepLR(optimizer,milestones=[10,20,30,40],gamma=0.5)
    num_steps = len(train_loader) * custom_num_epochs
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    print("Using CosineAnnealingLR+UntunedLinearWarmup")
    #############################################################

    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_acc_so_far = 0.000
    best_epoch_record = 0

    # The following two models ARE NOT TRAINABLE
    model_name = 'bert-base-chinese' if setting['train_dataname'] not in english_lists else 'bert-base-uncased'
    text_model = BertModel.from_pretrained(model_name).cuda()
    for param in text_model.parameters():
            param.requires_grad = False
    netG = torchvision.models.resnet50(pretrained=True)
    netG.fc = nn.Identity()
    netG = netG.cuda()

    similarity_module = SimilarityPart()
    similarity_module.cuda()
    rumor_module = MultiModal()
    rumor_module.cuda()

    loss_f_rumor = torch.nn.CrossEntropyLoss()
    loss_f_sim = torch.nn.CosineEmbeddingLoss()
    lr = 1e-3
    l2 = 0  # 1e-5
    optim_task1 = torch.optim.Adam(
        similarity_module.parameters(), lr=lr, weight_decay=l2)
    optim_task2 = torch.optim.Adam(
        rumor_module.parameters(), lr=lr, weight_decay=l2)

    loss_task1_total = 0
    loss_task2_total = 0
    best_acc = 0

    print('training model')
    for epoch in range(custom_num_epochs):
        # p = float(epoch) / 100
        # lr = 0.001 / (1. + 10 * p) ** 0.75
        # optimizer.lr = lr

        cost_vector = []
        acc_vector = []

        # puretext = puretext_iter.__next__()
        # pureimage = pureimage_iter.__next__()

        total = len(train_dataset)
        progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
        for i, items in enumerate(train_loader):

            with torch.enable_grad():
                logs = []

                texts, image, labels, category = items
                input_ids, attention_mask, token_type_ids = texts
                input_ids, attention_mask, token_type_ids, image, labels, category = \
                    to_var(input_ids), to_var(attention_mask), to_var(token_type_ids), \
                    to_var(image), to_var(labels), to_var(category)
                imageclip, textclip = None, None

                text = text_model(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids)[0]

                similarity_module.train()
                rumor_module.train()
                optim_task1.zero_grad()
                optim_task2.zero_grad()

                image = netG(image)
                text, image = text.clone().detach(), image.clone().detach()
                fixed_text, matched_image, unmatched_image = prepare_task2_data(text, image, labels)
                fixed_text = fixed_text.cuda()
                matched_image = matched_image.cuda()
                unmatched_image = unmatched_image.cuda()

                text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(fixed_text,
                                                                                                   matched_image)
                text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(fixed_text,
                                                                                                         unmatched_image)
                similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)
                similarity_label_0 = torch.cat(
                    [torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])],
                    dim=0).cuda()
                similarity_label_1 = torch.cat(
                    [torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])],
                    dim=0).cuda()

                text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
                image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
                loss_similarity = loss_f_sim(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)

                optim_task1.zero_grad()
                loss_similarity.backward()
                optim_task1.step()
                # task1中预测正确的个数
                # corrects_pre_sim += similarity_pred.eq(similarity_label_0).sum().item()

                # 再训练task2
                text_aligned, image_aligned, _ = similarity_module(text, image)
                pre_rumor = rumor_module(text, image, text_aligned, image_aligned)
                loss_rumor = loss_f_rumor(pre_rumor, labels)

                optim_task2.zero_grad()
                loss_rumor.backward()
                optim_task2.step()
                # task2预测正确的个数
                pre_label_rumor = pre_rumor.argmax(1)
                accuracy = pre_label_rumor.eq(labels.view_as(pre_label_rumor)).sum().item()/custom_batch_size

                logs.append(('CE_loss', loss_rumor.item()))


                cost_vector.append(loss_rumor.item())
                acc_vector.append(accuracy)
                mean_cost, mean_acc = np.mean(cost_vector), np.mean(acc_vector)
                # logs.append(('mean_cost', mean_cost))
                logs.append(('mean_acc', mean_acc))

                progbar.add(len(image), values=logs)


        print('Epoch [%d/%d],  Loss: %.4f, Train_Acc: %.4f,  '
              % (
                  epoch + 1, custom_num_epochs, np.mean(cost_vector), np.mean(acc_vector)))
        print("end training...")
        # with warmup_scheduler.dampening():
        #     scheduler.step()  # val_loss

        # test
        with torch.no_grad():
            total = len(validate_dataset)
            progbar = Progbar(total, width=10, stateful_metrics=stateful_metrics)
            for i, items in enumerate(validate_loader):
                logs = []

                texts, image, labels, category = items
                input_ids, attention_mask, token_type_ids = texts
                input_ids, attention_mask, token_type_ids, image, labels, category = \
                    to_var(input_ids), to_var(attention_mask), to_var(token_type_ids), \
                    to_var(image), to_var(labels), to_var(category)
                imageclip, textclip = None, None

                text = text_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

                similarity_module.train()
                rumor_module.train()
                optim_task1.zero_grad()
                optim_task2.zero_grad()

                image = netG(image)
                text, image = text.clone().detach(), image.clone().detach()
                fixed_text, matched_image, unmatched_image = prepare_task2_data(text, image, labels)
                fixed_text.cuda()
                matched_image.cuda()
                unmatched_image.cuda()

                text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(fixed_text,
                                                                                                   matched_image)
                text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(fixed_text,
                                                                                                         unmatched_image)
                similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)],
                                            dim=0)
                similarity_label_0 = torch.cat(
                    [torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])],
                    dim=0).cuda()
                similarity_label_1 = torch.cat(
                    [torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])],
                    dim=0).cuda()

                text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
                image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
                loss_similarity = loss_f_sim(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)

                # task1中预测正确的个数
                # corrects_pre_sim += similarity_pred.eq(similarity_label_0).sum().item()

                # 再训练task2
                text_aligned, image_aligned, _ = similarity_module(text, image)
                pre_rumor = rumor_module(text, image, text_aligned, image_aligned)
                loss_rumor = loss_f_rumor(pre_rumor, labels)

                # task2预测正确的个数
                pre_label_rumor = pre_rumor.argmax(1)
                accuracy = pre_label_rumor.eq(labels.view_as(pre_label_rumor)).sum().item()/custom_batch_size

                logs.append(('CE_loss', loss_rumor.item()))

                cost_vector.append(loss_rumor.item())
                acc_vector.append(accuracy)
                mean_cost, mean_acc = np.mean(cost_vector), np.mean(acc_vector)
                # logs.append(('mean_cost', mean_cost))
                logs.append(('mean_acc', mean_acc))

                progbar.add(len(image), values=logs)


            if accuracy > best_acc_so_far:
                best_acc_so_far = accuracy
                best_epoch_record = epoch + 1

            print('Epoch [%d/%d],  Val_Acc: %.4f. at thresh %.4f (so far %.4f in Epoch %d) .'
                  % (
                      epoch + 1, custom_num_epochs, accuracy, 0, best_acc_so_far, best_epoch_record,
                  ))
            # print("------Real News -----------")
            # print("Precision: {}".format(validate_real_precision))
            # print("Recall: {}".format(validate_real_recall))
            # print("Accuracy: {}".format(validate_real_accuracy))
            # print("F1: {}".format(validate_real_F1))
            # print("------Fake News -----------")
            # print("Precision: {}".format(validate_fake_precision))
            # print("Recall: {}".format(validate_fake_recall))
            # print("Accuracy: {}".format(validate_fake_accuracy))
            # print("F1: {}".format(validate_fake_F1))
            # print("---------------------------")
            print("end evaluate...")
            # if validate_acc > best_validate_acc:
            #     best_validate_acc = validate_acc
            #     if not os.path.exists(args.output_file):
            #         os.mkdir(args.output_file)
            #     best_validate_dir = "{}/{}/{}_{}{}_{}.pkl".format(args.output_file, setting['train_dataname'],
            #                                                       str(epoch + 1), str(datetime.datetime.now().month),
            #                                                       str(datetime.datetime.now().day),
            #                                                       int(best_validate_acc * 100))
            #     torch.save(model.state_dict(), best_validate_dir)
            #     print("Model saved at {}".format(best_validate_dir))


from collections import OrderedDict
def load_model(model, load_path, strict=True):
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=strict)


def parse_arguments(parser):
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    # parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')
    parser.add_argument('--dataset', type=str, default='weibo', help='')

    parse.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=25, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=512, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=25, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')

    #    parser.add_argument('--sequence_length', type = int, default = 28, help = '')
    #    parser.add_argument('--input_size', type = int, default = 28, help = '')
    #    parser.add_argument('--hidden_size', type = int, default = 128, help = '')
    #    parser.add_argument('--num_layers', type = int, default = 2, help = '')
    #    parser.add_argument('--num_classes', type = int, default = 10, help = '')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--num_epochs', type=int, default=50, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')

    #    args = parser.parse_args()
    return parser


def load_data(args, dataset):
    if dataset == 'weibo':
        import process_data_weibo as process_data
        train, validate = process_data.get_data(args.text_only)
    else:  # "Twitter"
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
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = ''
    test = ''
    output = '/home/groupshare/mae-main/output_dir/'
    args = parser.parse_args([train, test, output])

    main(args)


