# encoding=utf-8
import pickle as pickle
import random
from random import *
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
from collections import defaultdict
import sys, re
import pandas as pd
from PIL import Image
import math
from types import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path
import pandas
import sys
import importlib
from transformers import BertTokenizer

importlib.reload(sys)


#


def read_image():
    image_list = {}
    file_list = ['/home/groupshare/twitter-merged/']
    for path in file_list:
        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                # im = 1
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print(filename)
    print("image length " + str(len(image_list)))
    # print("image names are " + str(image_list.keys()))
    return image_list


def write_data(flag, image, text_only):
    def read_post_train(flag):
        if flag == "train":
            path = "/home/groupshare/mae-main/ztrain_en.xlsx"
            f = pandas.read_excel(path, usecols=[1, 2, 3, 4, 5])
        elif flag == "validate":
            path = "/home/groupshare/mae-main/ztest_en.xlsx"
            f = pandas.read_excel(path, usecols=[1, 2, 3, 4, 5])

        post_content = []
        data = []
        column = ['post_id', 'image_id', 'post_text', 'label', 'tran_texts']

        twitter_id = 0
        line_data = []
        cop1 = re.compile("[^a-z^A-Z^0-9^,^.^!^:^? ]")
        cop2 = re.compile("[/-@#]")
        cop = re.compile("[,.@#&/…)(;:'*!?a-zA-Z0-9 ]")

        for i in range(len(f) - 1):
            line_data = []
            twitter_id = str(f.iloc[i, 0])
            line_data.append(twitter_id)
            line_data.append(f.iloc[i, 2])
            t = str(f.iloc[i, 1]).split('http')[0]

            if len(cop.sub('', str(t))) < 5:
                post_content.append(cop1.sub('', cop2.sub(' ', str(t))))
                line_data.append(cop1.sub('', cop2.sub(' ', str(t))))
                line_data.append(f.iloc[i, 3])
                line_data.append(cop1.sub('', cop2.sub(' ', str(f.iloc[i, 4]))))

                data.append(line_data)
        #    else:
        #        print(str(t))

        # print(data)
        #     return post_content

        data_df = pd.DataFrame(np.array(data), columns=column)

        return post_content, data_df

    post_content, post = read_post_train(flag)

    print("Original post length is " + str(len(post_content)))
    print("Original data frame is " + str(post.shape))

    def paired(text_only=False):
        ordered_image = []
        ordered_post = []
        trans_post = []
        label = []
        post_id = []
        image_id_list = []
        # image = []

        image_id = ""
        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split(','):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in image:
                    break
            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])
                ordered_post.append(post.iloc[i]['post_text'])
                trans_post.append(post.iloc[i]['tran_texts'])
                post_id.append(id)

                label.append(post.iloc[i]['label'])

        label = np.array(label, dtype=np.int)

        print("Label number is " + str(len(label)))
        print("Rummor number is " + str(sum(label)))
        print("Non rummor is " + str(len(label) - sum(label)))

        #
        if flag == "test":
            y = np.zeros(len(ordered_post))
        else:
            y = []

        data = {"post_text": np.array(ordered_post),
                "image": ordered_image, "social_feature": [],
                "label": np.array(label), \
                "image_id": image_id_list,
                "tran_texts": np.array(trans_post)}
        # print(data['image'][0])

        print("data size is " + str(len(data["post_text"])))

        return data

    paired_data = paired(text_only)

    print("paired post length is " + str(len(paired_data["post_text"])))
    print("paried data has " + str(len(paired_data)) + " dimension")
    return paired_data


def get_data(text_only):
    # text_only = False

    if text_only:
        print("Text only")
        image_list = []
    else:
        print("Text and image")
        image_list = read_image()

    train_data = write_data("train", image_list, text_only)
    validate_data = write_data("validate", image_list, text_only)
    # test_data = write_data("test", image_list, text_only)

    # f = open('train.pckl','wb')
    # pickle.dump(train_data, f)
    # f.close()
    # f = open('validate.pckl','wb')
    # pickle.dump(validate_data, f)
    # f.close()
    # f = open('test.pckl','wb')
    # pickle.dump(test_data, f)
    # f.close()

    print("loading data...")
    # print(str(len(all_text)))

    # return train_data, validate_data, test_data
    return train_data, validate_data

# if __name__ == "__main__":
#     image_list = read_image()
#
#     train_data = write_data("train", image_list)
#     valiate_data = write_data("validate", image_list)
#     test_data = write_data("test", image_list)
#
#     # print("loading data...")
#     # # w2v_file = '/data/EANN-KDD18-3w/data/GoogleNews-vectors-negative300.bin'
#     vocab, all_text = load_data(train_data, test_data)
#     #
#     # # print(str(len(all_text)))
#     #
#     # print("number of sentences: " + str(len(all_text)))
#     # print("vocab size: " + str(len(vocab)))
#     # max_l = len(max(all_text, key=len))
#     # print("max sentence length: " + str(max_l))
#     #
#     # #
#     # #
#     # word_embedding_path = "/data/EANN-KDD18-3w/data/weibo/word_embedding.pickle"
#     # if not os.path.exists(word_embedding_path):
#     #     min_count = 1
#     #     size = 32
#     #     window = 4
#     #
#     #     w2v = Word2Vec(all_text, min_count=min_count, size=size, window=window)
#     #
#     #     temp = {}
#     #     for word in w2v.wv.vocab:
#     #         temp[word] = w2v[word]
#     #     w2v = temp
#     #     pickle.dump(w2v, open(word_embedding_path, 'wb+'))
#     # else:
#     #     w2v = pickle.load(open(word_embedding_path, 'rb'))
#     # # print(temp)
#     # # #
#     # print("word2vec loaded!")
#     # print("num words already in word2vec: " + str(len(w2v)))
#     # # w2v = add_unknown_words(w2v, vocab)
#     # Whole_data = {}
#     # file_path = "/data/EANN-KDD18-3w/data/weibo/event_clustering.pickle"
#     # # if not os.path.exists(file_path):
#     # #     data = []
#     # #     for l in train_data["post_text"]:
#     # #         line_data = []
#     # #         for word in l:
#     # #             line_data.append(w2v[word])
#     # #         line_data = np.matrix(line_data)
#     # #         line_data = np.array(np.mean(line_data, 0))[0]
#     # #         data.append(line_data)
#     # #
#     # #     data = np.array(data)
#     # #
#     # #     cluster = AgglomerativeClustering(n_clusters=15, affinity='cosine', linkage='complete')
#     # #     cluster.fit(data)
#     # #     y = np.array(cluster.labels_)
#     # #     pickle.dump(y, open(file_path, 'wb+'))
#     # # else:
#     # # y = pickle.load(open(file_path, 'rb'))
#     # # print("Event length is " + str(len(y)))
#     # # center_count = {}
#     # # for k, i in enumerate(y):
#     # #     if i not in center_count:
#     # #         center_count[i] = 1
#     # #     else:
#     # #         center_count[i] += 1
#     # # print(center_count)
#     # # train_data['event_label'] = y
#     #
#     # #
#     # print("word2vec loaded!")
#     # print("num words already in word2vec: " + str(len(w2v)))
#     # add_unknown_words(w2v, vocab)
#     # W, word_idx_map = get_W(w2v)
#     # # # rand_vecs = {}
#     # # # add_unknown_words(rand_vecs, vocab)
#     # W2 = rand_vecs = {}
#     # pickle.dump([W, W2, word_idx_map, vocab, max_l], open("/data/EANN-KDD18-3w/data/weibo/word_embedding.pickle", "wb"))
#     # print("dataset created!")



