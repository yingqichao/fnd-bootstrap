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
#import clip
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

# clipmodel, preprocess = clip.load('ViT-B/32', device)
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# translator = Translator(service_urls=[
#     'translate.google.cn'
# ])

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # torch.cuda._initialized = True
    # torch.backends.cudnn.benchmark = True
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    print("world: {},rank: {},num_gpus:{}".format(world_size,rank,num_gpus))
    return world_size, rank


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

    if len(item) <= 2:
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

    if len(item) <= 2:
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
    setting['val'] = args.val
    setting['is_filter'] = False
    setting['duplicate_fake_times'] = args.duplicate_fake_times
    setting['is_use_unimodal'] = True
    setting['with_ambiguity'] = True
    # CURRENTLY ONLY SUPPORT GOSSIP Weibo
    LIST_ALLOW_AMBIGUITY = ['gossip','weibo']
    setting['with_ambiguity'] = setting['with_ambiguity'] and setting['train_dataname'] in LIST_ALLOW_AMBIGUITY
    setting['data_augment'] = False
    setting['is_use_bce'] = True
    setting['use_soft_label'] = False
    setting['is_sample_positive'] = args.is_sample_positive #if setting['train_dataname'] != 'gossip' else 0.3
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

    ########## validate dataset ####################
    if setting['val_dataname']=='weibo':
        print("Using weibo as inference")
        from data.weibo_dataset_test_modification import weibo_dataset_test
        validate_dataset = weibo_dataset_test()
        validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False,
                                     collate_fn=collate_fn_chinese,
                                     num_workers=1, sampler=None, drop_last=False,
                                     pin_memory=True)

    # elif setting['val_dataname']=='Twitter':
    #     from data.Twitter_dataset import Twitter_dataset
    #     print("Using Twitter as inference")
    #     validate_dataset = Twitter_dataset(is_train=False, image_size=GT_size)
    #     validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,
    #                                  collate_fn=collate_fn_english,
    #                                  num_workers=4, sampler=None, drop_last=True,
    #                                  pin_memory=True)
    #
    # elif setting['val_dataname']=='Mix':
    #     from data.MixSet_dataset import MixSet_dataset
    #     print("using Mix as inference")
    #     validate_dataset = MixSet_dataset(is_train=False, image_size=GT_size)
    #     validate_loader = DataLoader(validate_dataset, batch_size=custom_batch_size, shuffle=False,collate_fn=collate_fn_chinese,
    #                                        num_workers=4, sampler=None, drop_last=True,
    #                                        pin_memory=True)
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
    # if is_use_WWW_loader:
    #     from models.UAMFDforWWW_Net import UAMFD_Net
    #     model = UAMFD_Net(dataset=setting['train_dataname'],is_use_bce=setting['is_use_bce'])
    # else:
    # from models.UAMFD_Net import UAMFD_Net
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

    print(get_parameter_number(model))
    ############################################################
    ##################### Loss and Optimizer ###################
    loss_cross_entropy = nn.CrossEntropyLoss().cuda()
    loss_focal = focal_loss(alpha=0.25, gamma=2, num_classes=2).cuda()
    loss_bce = nn.BCEWithLogitsLoss().cuda()
    criterion = loss_bce if setting['is_use_bce'] else loss_focal

    num_steps = int(len(train_loader) * custom_num_epochs * 1.2)

    print("Using CosineAnnealingLR+UntunedLinearWarmup")
    #############################################################

    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_acc_so_far = 0.000
    best_epoch_record = 0
    global_step = 0
    print('training model')

    model.eval()
    validate_acc_vector_temp, validate_precision_vector_temp, validate_recall_vector_temp, validate_F1_vector_temp  = [], [], [], []
    val_loss = 0
    ## THRESH: 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 ##
    THRESH = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    realnews_TP, realnews_TN, realnews_FP, realnews_FN = [0]*9, [0]*9, [0]*9, [0]*9
    fakenews_TP, fakenews_TN, fakenews_FP, fakenews_FN = [0]*9, [0]*9, [0]*9, [0]*9
    realnews_sum, fakenews_sum = [0]*9, [0]*9
    img_correct, text_correct, vgg_correct = [0]*9, [0]*9, [0]*9
    y_pred_full, y_GT_full = None, None
    y_pred_fake_full, y_GT_fake_full, y_pred_real_full, y_GT_real_full = None, None, None, None
    image_no,results = 0,[]
    dataset_name = setting['val_dataname']
    for i, items in enumerate(validate_loader):
        # if setting['train_dataname'] == 'WWW':
        #     ####### WWW DEPRECATED #############
        #     input_ids, image, labels = items
        #     # input_ids, image, labels = to_var(input_ids), to_var(image), to_var(labels)
        #     attention_mask, token_type_ids, imageclip, textclip = None, None, None, None
        # # elif setting['with_ambiguity']:
        # #     texts, image, labels, category, texts1, image1, labels1 = items
        # #     input_ids, attention_mask, token_type_ids = texts
        # #     input_ids1, attention_mask1, token_type_ids1 = texts1
        # #     input_ids, attention_mask, token_type_ids, image, labels, category = \
        # #         to_var(input_ids), to_var(attention_mask), to_var(token_type_ids), \
        # #         to_var(image), to_var(labels), to_var(category)
        # #     input_ids1, attention_mask1, token_type_ids1, image1, labels1 = \
        # #         to_var(input_ids1), to_var(attention_mask1), to_var(token_type_ids1), \
        # #         to_var(image1), to_var(labels1)
        # else:
        texts, others, GT_path = items
        input_ids, attention_mask, token_type_ids = texts
        image, labels, category, sents = others
        input_ids, attention_mask, token_type_ids, image, labels, category = \
            to_var(input_ids), to_var(attention_mask), to_var(token_type_ids), \
            to_var(image), to_var(labels), to_var(category)

        mix_output, image_only_output, text_only_output, vgg_only_output, aux_output, _ = model(input_ids=input_ids,
                                                                        attention_mask=attention_mask,
                                                                        token_type_ids=token_type_ids,
                                                                        image=image,
                                                                        no_ambiguity=True,
                                                                        category=category)
        # _, argmax = torch.max(Mix_output, 1)
        # vali_loss = criterion(validate_outputs, labels)
        if setting['is_use_bce']:
            # mix_output = mix_output[:, :-1]
            labels = labels.float().unsqueeze(1)

        val_loss = criterion(mix_output, labels)
        val_img_loss = criterion(image_only_output, labels)
        val_text_loss = criterion(text_only_output, labels)
        val_vgg_loss = criterion(vgg_only_output, labels)
        if progbar is not None:
            logs = []
            logs.append(('mix_loss', val_loss.item()))
            logs.append(('image_loss', val_img_loss.item()))
            logs.append(('text_loss', val_text_loss.item()))
            logs.append(('vgg_loss', val_vgg_loss.item()))
            progbar.add(len(image), values=logs)

        mix_output, image_only_output, text_only_output, vgg_only_output, aux_output = torch.sigmoid(mix_output), torch.sigmoid(
            image_only_output), torch.sigmoid(text_only_output), torch.sigmoid(vgg_only_output), torch.sigmoid(aux_output)

        for thresh_idx, thresh in enumerate(THRESH):
            # _, validate_argmax = torch.max(validate_outputs, 1)
            validate_argmax = torch.where(mix_output<thresh,0,1)
            validate_img_argmax = torch.where(image_only_output < thresh, 0, 1)
            validate_text_argmax = torch.where(text_only_output < thresh, 0, 1)
            validate_vgg_argmax = torch.where(vgg_only_output < thresh, 0, 1)
            y_pred = validate_argmax.squeeze().cpu().numpy() #y_pred = torch.tensor([0, 1, 0, 0])
            y_pred_img = validate_img_argmax.squeeze().cpu().numpy()
            y_pred_text = validate_text_argmax.squeeze().cpu().numpy()
            y_pred_vgg = validate_vgg_argmax.squeeze().cpu().numpy()
            y_GT = labels.int().cpu().numpy() #y_true=torch.tensor([0, 1, 0, 1])

            for idx, _ in enumerate(y_pred):
                if thresh_idx==0:
                    record = {}
                    record['image_no'], record['text'] = image_no, sents[idx]
                    record['y_GT'], record['y_pred'] = y_GT[idx], mix_output[idx].item()
                    record['y_pred_mm'], record['y_pred_img'], record['y_pred_text'], record['y_pred_vgg'] = aux_output[idx].item(), \
                                                                                                             image_only_output[idx].item(), \
                                                                                                             text_only_output[idx].item(), \
                                                                                                             vgg_only_output[idx].item()
                    # soft_scores = torch.softmax(
                    #     torch.cat((aux_output[idx], image_only_output[idx], text_only_output[idx], vgg_only_output[idx]), dim=0), dim=0)
                    # record['soft_mm'], record['soft_img'], record['soft_text'], record['soft_vgg'] = soft_scores[0].item(), \
                    #                                                                     soft_scores[1].item(), \
                    #                                                                     soft_scores[2].item(), \
                    #                                                                     soft_scores[3].item()

                    save_name = f'/home/groupshare/mae-main/example/{dataset_name}/{image_no}.png'
                    if not os.path.exists(save_name):
                        torchvision.utils.save_image((image[idx:idx+1] * 255).round() / 255,
                                                     save_name, nrow=1, padding=0, normalize=False)
                    results.append(record)
                    image_no += 1

                if y_pred_img[idx]==y_GT[idx]: img_correct[thresh_idx] += 1
                if y_pred_text[idx] == y_GT[idx]: text_correct[thresh_idx] += 1
                if y_pred_vgg[idx] == y_GT[idx]: vgg_correct[thresh_idx] += 1

                if y_GT[idx]==1:
                    #  FAKE NEWS RESULT
                    fakenews_sum[thresh_idx] +=1
                    if y_pred[idx]==0:
                        fakenews_FN[thresh_idx] += 1
                        realnews_FP[thresh_idx] += 1
                    else:
                        fakenews_TP[thresh_idx] += 1
                        realnews_TN[thresh_idx] += 1
                else:
                    # REAL NEWS RESULT
                    realnews_sum[thresh_idx] +=1
                    if y_pred[idx]==1:
                        realnews_FN[thresh_idx] +=1
                        fakenews_FP[thresh_idx] +=1
                    else:
                        realnews_TP[thresh_idx] += 1
                        fakenews_TN[thresh_idx] += 1
            # val_accuracy[thresh_idx] = metrics.accuracy_score(y_GT, y_pred,pos_label=1,average='binary',sample_weight=None)
            # real_precision[thresh_idx] = metrics.precision_score(y_GT, y_pred)
            # real_recall[thresh_idx] = metrics.recall_score(y_GT, y_pred)
            # real_accuracy[thresh_idx] = metrics.accuracy_score(y_GT, y_pred)
            # real_F1[thresh_idx] = metrics.f1_score(y_GT, y_pred)
            # fake_precision[thresh_idx] = metrics.precision_score(y_GT, y_pred)
            # fake_recall[thresh_idx] = metrics.recall_score(y_GT, y_pred)
            # fake_accuracy[thresh_idx] = metrics.accuracy_score(y_GT, y_pred)
            # fake_F1[thresh_idx] = metrics.f1_score(y_GT, y_pred)
    import pandas as pd
    df = pd.DataFrame(results)
    pandas_file = f'/home/groupshare/mae-main/example/{dataset_name}/experiment.xlsx'
    df.to_excel(pandas_file)
    print(f"Excel Saved at {pandas_file}")

    val_accuracy, real_accuracy, fake_accuracy, real_precision, fake_precision = [0]*9,[0]*9,[0]*9,[0]*9,[0]*9
    real_recall, fake_recall, real_F1, fake_F1 = [0]*9,[0]*9,[0]*9,[0]*9
    for thresh_idx, _ in enumerate(THRESH):
        img_correct[thresh_idx] = img_correct[thresh_idx]/(realnews_sum[thresh_idx]+fakenews_sum[thresh_idx])
        text_correct[thresh_idx] = text_correct[thresh_idx] / (realnews_sum[thresh_idx] + fakenews_sum[thresh_idx])
        vgg_correct[thresh_idx] = vgg_correct[thresh_idx] / (realnews_sum[thresh_idx] + fakenews_sum[thresh_idx])

        val_accuracy[thresh_idx] = (realnews_TP[thresh_idx]+realnews_TN[thresh_idx])/(realnews_TP[thresh_idx]+realnews_TN[thresh_idx]+realnews_FP[thresh_idx]+realnews_FN[thresh_idx])
        real_accuracy[thresh_idx] = (realnews_TP[thresh_idx])/realnews_sum[thresh_idx]
        fake_accuracy[thresh_idx] = (fakenews_TP[thresh_idx])/fakenews_sum[thresh_idx]
        real_precision[thresh_idx] = realnews_TP[thresh_idx]/max(1,(realnews_TP[thresh_idx]+realnews_FP[thresh_idx]))
        fake_precision[thresh_idx] = fakenews_TP[thresh_idx] / max(1,(fakenews_TP[thresh_idx] + fakenews_FP[thresh_idx]))
        real_recall[thresh_idx] = realnews_TP[thresh_idx]/max(1,(realnews_TP[thresh_idx]+realnews_FN[thresh_idx]))
        fake_recall[thresh_idx] = fakenews_TP[thresh_idx] / max(1,(fakenews_TP[thresh_idx] + fakenews_FN[thresh_idx]))
        real_F1[thresh_idx] = 2*(real_recall[thresh_idx]*real_precision[thresh_idx])/max(1,(real_recall[thresh_idx]+real_precision[thresh_idx]))
        fake_F1[thresh_idx] = 2 * (fake_recall[thresh_idx] * fake_precision[thresh_idx]) / max(1,(fake_recall[thresh_idx] + fake_precision[thresh_idx]))

    return val_accuracy, (real_precision, real_recall, real_accuracy, real_F1),\
           (fake_precision, fake_recall, fake_accuracy, fake_F1), \
           val_loss,\
           (img_correct,text_correct,vgg_correct)


def load_data(args, dataset):
    if dataset=='weibo':
        import process_data_weibo as process_data
        train, validate = process_data.get_data(args.text_only)
    else: # "Twitter"
        import process_data_Twitter as process_data
        train, validate = process_data.get_data(args.text_only)

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
    parser.add_argument('-val', type=int, default=0, help='')
    parser.add_argument('-duplicate_fake_times', type=int, default=1, help='')
    parser.add_argument('-is_sample_positive', type=float, default=1.0, help='')
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


