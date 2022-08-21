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

# from imagededup.methods import PHash
# phasher = PHash()

from variables import rumor_root, weibo_fake_root, weibo_real_root, weibo21_fake_root, weibo21_real_root, \
    get_workbook, del_emoji, strQ2B

def truncate_tweet():
    # df_list = [
    #     'In Borno/Yobe/Lake Chad?? RT \"@abati1990: Nigeria\'s committed &amp;patriotic troops in action to #BringBackOurGirls.',
    #     'God bless these patriots! \"@abati1990: Nigeria\'s committed &amp;patriotic troops in action to #BringBackOurGirls.',
    #     '#GodblessNigArmy \"@abati1990: Nigeria\'s committed #&amp;patriotic troops in action to #BringBackOurGirls.',
    #     'Got Sochi Problems? Russians say: \"Stop moaning and enjoy the games!\" http://t.co/BuF4sn904m',
    #     'Is that real?? Whoa RT @pablogarabito: #Sandy',
    #     'Is this 4 real? #sandy',
    #     'This shit is serious!! I hope everyone is safe #sandy #subway #under #water'
    #     ]
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
    # vec.fit(df_list)
    # features = vec.transform(df_list)
    #
    # from sklearn.metrics.pairwise import cosine_similarity
    # cosine_sim = cosine_similarity(features)
    # similar = list(enumerate(cosine_sim[0]))
    # print(similar)

    num_truncated, num_allowed = 0,3
    topic_by_images = {}
    xlsx = "./dataset/Twitter/train_twitter.xlsx"
    sheet_rumor, rows_rumor = get_workbook(xlsx)
    records_list = []
    indices = [i for i in range(2, rows_rumor + 1)]
    # image label content: E H C
    while len(indices)>=0:  # title label content image B C E F
        i = random.randint(0,len(indices)+1)
        # NOTE: 1 REPRESENTS REAL NEWS
        records = {}
        tweetId = str(sheet_rumor['B' + str(i)].value)
        userId = str(sheet_rumor['D' + str(i)].value)
        images_name = str(sheet_rumor['E' + str(i)].value)
        user_name = str(sheet_rumor['F' + str(i)].value)
        timestamp = str(sheet_rumor['G' + str(i)].value)
        label = int(sheet_rumor['H' + str(i)].value)
        content = str(sheet_rumor['C' + str(i)].value)
        key = images_name+str(label)
        if key not in topic_by_images:
            topic_by_images[key] = []



        del indices[i]

    df = pd.DataFrame(records_list)
    df.to_excel('./dataset/Twitter/train_twitters_truncate.xlsx')

    print("sum: {} truncated{}".format(len(records_list),num_truncated))


def reload_tweet():

    num_dups, num_length, num_not_found, num_sandy, num_humor = 0, 0, 0, 0, 0
    mediaeval2015 = 'E:/Twitter/image-verification-corpus-master/mediaeval2015/devset/MediaEval2015_DevSet_tweets.txt'
    mediaeval2016 = 'E:/Twitter/image-verification-corpus-master/mediaeval2016/devset/MediaEval2016_DevSet_tweets.txt'
    mediaeval2015test = 'E:/Twitter/image-verification-corpus-master/mediaeval2015/testset/MediaEval2015_test_tweets.txt'
    mediaeval2016test = 'E:/Twitter/image-verification-corpus-master/mediaeval2016/testset/MediaEval2016_TestSet_posts_groundtruth.txt'
    ###### train ########
    # text_lists = [mediaeval2015, mediaeval2015test, mediaeval2016]
    # image_paths_train = ['E:/Twitter/images/images/']
    # mode = 'train'
    ###### test #########
    text_lists = [mediaeval2016test]
    image_paths_train = ['E:/Twitter/image-verification-corpus-master/mediaeval2016/testset/Mediaeval2016_TestSet_Images/Mediaeval2016_TestSet_Images/']
    mode = 'test'

    images, num_not_found_set, cannot_open_set = {}, set(), set()
    for image_path_train in image_paths_train:
        items = os.listdir(image_path_train)
        for item in items:
            idx = item.rfind('.')
            images[item[:idx]] = item

    used_images = copy.deepcopy(images)
    results, duplicated_tweets = [], {}
    for i in tqdm(range(len(text_lists))):
        text_list = text_lists[i]
        f = open(text_list, encoding='UTF-8')  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        line_num, results_dict = 0, {}
        while line:
            # if line_num==15626:
            #     here=1
            if line_num != 0:  # tweetId	tweetText	userId	imageId(s)	username	timestamp	label
                results_dict = {}
                is_found_dup, is_add_key = False, True
                items = line.split('\t')
                tweetText = items[1]
                http_idx = tweetText.rfind('http://')
                https_idx = tweetText.rfind('https://')
                idx = max(http_idx, https_idx)
                tweetText = tweetText[:idx].rstrip().lstrip()
                tweetText = del_emoji(tweetText)
                # tweetText = tweetText.replace('\t','').replace('\n','')
                if mode!='test' and len(tweetText)<20:
                    print("Found length not satisfy {}".format(tweetText))
                    num_length += 1
                    line = f.readline()
                    line_num += 1
                    continue

                imageIds = items[3 if text_list!=mediaeval2016test else 4].rstrip().lstrip()
                query_tweet = strQ2B(tweetText[5:100]).lower()
                # 先看一下能否打开
                is_found_image = False
                imageId, newimageIds = imageIds.split(','), ''
                for image_item in imageId:
                    is_found_image = False
                    if image_item in images:
                        img = images[image_item]
                        is_found_image = cv2.imread(image_paths_train[0]+img) is not None
                        if not is_found_image: cannot_open_set.add(img)
                        if img in used_images: del used_images[image_item]
                    if not is_found_image and image_item[:5]=='sandy':
                            image_item_A, image_item_B = 'sandyA'+image_item[5:], 'sandyB'+image_item[5:]
                            if image_item_A in images or image_item_B in images:
                                img = images[image_item_A if image_item_A in images else image_item_B]
                                is_found_image = cv2.imread(image_paths_train[0]+img) is not None
                                if not is_found_image: cannot_open_set.add(img)
                                if img in used_images: del used_images[img]
                                num_sandy +=1
                                print("Modify {} to {}".format(image_item,img))

                    if not is_found_image:
                        print("Error: not found {}".format(image_item))
                        num_not_found_set.add(image_item)
                    else:
                        newimageIds = newimageIds + image_item + ','

                if not is_found_image:
                    num_not_found += 1
                    line = f.readline()
                    line_num += 1
                    continue
                else:
                    imageIds = newimageIds[:-1]

                if mode!='test': # 测试集不去重
                    # 注意：有些twitter会在重复语句前面加上repost，所以需要做匹配
                    for target_tweet in duplicated_tweets:
                        queried_item = duplicated_tweets[target_tweet]
                        if query_tweet in target_tweet:
                            images_set = queried_item['image']
                            if imageIds not in images_set:
                                images_set.add(imageIds)
                                queried_item['image'] = images_set
                                is_add_key = False
                            else:
                                print("duplicated {} |{} {}[in] {}|{} {}/{}"
                                      .format(query_tweet, text_list,line_num,
                                              target_tweet, queried_item['text_list'], queried_item['line_num'],
                                              imageIds))
                                num_dups += 1
                                is_found_dup = True
                                break

                    if is_found_dup:
                        line = f.readline()
                        line_num += 1
                        continue
                    elif is_add_key:
                        key_dict = {}
                        images_set = set()
                        images_set.add(imageIds)
                        key_dict['image'] = images_set
                        key_dict['line_num'] = line_num
                        key_dict['text_list'] = text_list
                        duplicated_tweets[query_tweet] = key_dict

                results_dict['tweetId'] = items[0].rstrip().lstrip()
                results_dict['tweetText'] = tweetText
                results_dict['userId'] = items[2].rstrip().lstrip()
                results_dict['imageId(s)'] = imageIds
                results_dict['username'] = items[4 if text_list!=mediaeval2016test else 3].rstrip().lstrip()
                results_dict['timestamp'] = items[5].rstrip().lstrip()
                if 'real' in items[6]: results_dict['label'] = 1
                elif 'fake' in items[6]: results_dict['label'] = 0
                else:   results_dict['label'] = 2 # 2 stands for humor
                if results_dict['label']==2:
                    # skip humor
                    num_humor +=1
                else:
                    results.append(results_dict)
            line = f.readline()
            line_num += 1
        f.close()

    df = pd.DataFrame(results)
    df.to_excel('./dataset/Twitter/{}_datasets_WWW.xlsx'.format(mode))
    print("Records {} Dups {} Length {} NotFound {} Sandy {} Humor {}"
          .format(len(results),num_dups,num_length, num_not_found, num_sandy, num_humor))
    print(num_not_found_set)
    print(used_images)
    print(cannot_open_set)
