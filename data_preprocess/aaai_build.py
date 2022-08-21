import time
import json
from tkinter import *
from PIL import Image, ImageTk
import wget
import copy
import os
import openpyxl
import pandas as pd
import numpy as np
import paramiko
from scp import SCPClient
from tqdm import tqdm
import datetime
import random
from tkinter.filedialog import askdirectory, askopenfilename
import shutil
import options.options as option
import photohash
import cv2
import pickle
import random
random.seed(1000)

from variables import rumor_root, weibo_fake_root, weibo_real_root, weibo21_fake_root, weibo21_real_root, \
    get_workbook

from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
to_tensor = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),]
)

def generate_more_negative_AAAI():
    #   WE NEED TO SELECT 5000 ALIGNED IMAGE+TEXT AS WELL AS 5000 PAIRS NON-ALIGNED PAIRS

    dataset_name = 'gossip'
    xlsx = "./dataset/{}/{}_{}.xlsx".format(dataset_name, dataset_name, "train_no_filt")
    sheet_rumor, rows_rumor = get_workbook(xlsx)
    records_list = []
    valid_list_positive = [i for i in range(2,rows_rumor+1)]
    valid_list_negative = [i for i in range(2, rows_rumor + 1)]
    valid_list_negative_pool = [i for i in range(2, rows_rumor + 1)]
    num_real_count, num_fake_count = 0,0
    # GET 10000 PAIRS
    for iN in range(2, rows_rumor + 1):  # title label content image B C E F
        # NOTE: 1 REPRESENTS REAL NEWS
        records = {}

        images_name = str(sheet_rumor['C' + str(iN)].value)
        label = int(sheet_rumor['D' + str(iN)].value)
        content = str(sheet_rumor['B' + str(iN)].value)
        news_type = str(sheet_rumor['E' + str(iN)].value)
        if num_fake_count < 5000:
            if label == 0 and news_type != "image":
                iPool, label_pool, news_type_pool, images_name_pool = iN, 1, "text", ""
                while news_type_pool=="text" or iPool==iN:
                    idx_ranP = random.randint(0, len(valid_list_negative_pool)-1)
                    iPool = valid_list_negative_pool[idx_ranP]
                    images_name_pool = str(sheet_rumor['C' + str(iPool)].value)
                    label_pool = int(sheet_rumor['D' + str(iPool)].value)
                    news_type_pool = str(sheet_rumor['E' + str(iPool)].value)
                    content_pool = str(sheet_rumor['B' + str(iPool)].value)
                    if news_type_pool=="text" or iPool==iN:
                        valid_list_negative_pool.remove(iPool)
                records['content'] = content
                records['image'] = images_name_pool
                records['label'] = 0
                records['type'] = news_type
                records['debug'] = "{} {}".format(iN,iPool)
                records_list.append(records)
                records = {}
                valid_list_negative_pool.remove(iPool)
                num_fake_count += 1
            valid_list_negative.remove(iN)
    print("{} {} {}".format(num_real_count,num_fake_count,len(records_list)))
    ambiguity_excel = "./dataset/{}/{}_{}.xlsx".format(dataset_name, dataset_name, "more_negative")
    df = pd.DataFrame(records_list)
    df.to_excel(ambiguity_excel)


def reload_xlsxs_AAAI(dataset_name=['gossip']):
    root_path = 'E:/AAAI_dataset/AAAI_dataset'
    # dataset_name = ['gossip', 'politi']
    # fold_name = 'FakeNewsNet'
    fold_name = dataset_name[0]
    sub_folders = ['train','test']
    num_invalid_format, num_too_small, num_invalid_hashing, num_length = 0, 0, 0, 0
    hashing_not_allowed, ban_images = {}, os.listdir('E:/AAAI_dataset/AAAI_dataset/Images/ban_images')
    for ban_image in ban_images:
        hash = photohash.average_hash('E:/AAAI_dataset/AAAI_dataset/Images/ban_images/' + ban_image)
        hashing_not_allowed[ban_image] = hash

    for dataset in dataset_name:
        for sub_folder in sub_folders:
            xlsx = "{}/{}_{}.xlsx".format(root_path,dataset,sub_folder)
            sheet_rumor, rows_rumor = get_workbook(xlsx)
            records_list = []
            for i in tqdm(range(2, rows_rumor + 1)):  # title label content image B C E F
                records, news_type = {}, "multi"
                images_name = str(sheet_rumor['C' + str(i)].value)
                label = int(sheet_rumor['D' + str(i)].value)
                content = str(sheet_rumor['B' + str(i)].value)
                image_full_path = "{}/Images/{}_{}/{}".format(root_path,dataset,sub_folder,images_name)
                if len(content)<15:
                    news_type = "image"
                    print("Length not enough {} skip..".format(image_full_path))
                    num_length += 1
                    # if not NO_FILTER_OUT or label==1:
                    continue

                image_open = cv2.imread(image_full_path)
                if image_open is None:
                    image_open = Image.open(image_full_path)
                    image_tensor = to_tensor(image_open)
                    if image_open is None:
                        print("PIL still cannot open {} skip..".format(image_full_path))
                        num_invalid_format += 1
                        continue
                    image_width, image_height = image_tensor.shape[1], image_tensor.shape[2]
                else:
                    image_width, image_height = image_open.shape[0], image_open.shape[1]

                ## CONDUCT FILTERING OUT INVALID NEWS IF NOT NO_FILTER_OUT
                # IMAGE SIZE

                if image_width<100 or image_height<100:
                    news_type = "text"
                    print("Size too small {} skip..".format(image_full_path))
                    num_too_small += 1
                    # if not NO_FILTER_OUT or label==1:
                    continue
                # IMAGE HASHING
                found_invalid_hashing = False
                item1_hash = photohash.average_hash(image_full_path)
                for key in hashing_not_allowed:
                    item2_hash = hashing_not_allowed[key]
                    if item1_hash is not None and photohash.hashes_are_similar(item1_hash, item2_hash, tolerance=0.5):
                        found_invalid_hashing = True
                        break
                if found_invalid_hashing:
                    news_type = "text"
                    print("Invalid image found {} skip..".format(image_full_path))
                    num_invalid_hashing += 1
                    # if not NO_FILTER_OUT or label==1:
                    continue
                records['content'] = content
                records['image'] = images_name
                records['label'] = label
                ##  NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
                records['type'] = news_type
                records_list.append(records)

            df = pd.DataFrame(records_list)
            df.to_excel('./dataset/{}/origin_do_not_modify/{}_{}.xlsx'.format(fold_name, dataset,sub_folder))
    print("num_invalid_format, num_too_small, num_invalid_hashing, num_word_len {} {} {}"
          .format(num_invalid_format, num_too_small, num_invalid_hashing, num_length))

if __name__ == '__main__':
    reload_xlsxs_AAAI()
