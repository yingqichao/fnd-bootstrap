from torch.utils.data import DataLoader
from dataset import FeatureDataSet
from Mrmu_1010 import SimilarityPart, MultiModal
# from config import *
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.metrics import roc_curve, auc
# import visdom
# import matplotlib.pylab as plt
import copy
from torch.autograd import Variable
from tqdm import tqdm


def prepare_task2_data(text, image, label):
    nr_index = [i for i, la in enumerate(label) if la == 1]
    text_nr = text[nr_index]
    image_nr = image[nr_index]
    fixed_text = copy.deepcopy(text_nr)
    matched_image = copy.deepcopy(image_nr)
    unmatched_image = copy.deepcopy(image_nr).roll(shifts=3, dims=0)
    return fixed_text, matched_image, unmatched_image


def train():
    device = torch.device("cuda:0")
    batch_size = 1
    lr = 1e-3
    l2 = 0  # 1e-5
    NUM_WORKER = 1

    dataset_dir = './'

    train_set = FeatureDataSet(
        "{}/train_text+label.npz".format(dataset_dir),
        "{}/train_image+label.npz".format(dataset_dir), )

    test_set = FeatureDataSet(
        "{}/test_text+label.npz".format(dataset_dir),
        "{}/test_image+label.npz".format(dataset_dir), )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=NUM_WORKER,
        shuffle=True)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=NUM_WORKER)

    similarity_module = SimilarityPart()
    similarity_module.to(device)
    rumor_module = MultiModal()
    rumor_module.to(device)

    loss_f_rumor = torch.nn.CrossEntropyLoss()
    loss_f_sim = torch.nn.CosineEmbeddingLoss()

    optim_task1 = torch.optim.Adam(
        similarity_module.parameters(), lr=lr, weight_decay=l2)
    optim_task2 = torch.optim.Adam(
        rumor_module.parameters(), lr=lr, weight_decay=l2)

    loss_task1_total = 0
    loss_task2_total = 0
    best_acc = 0

    for epoch in range(100):

        similarity_module.train()
        rumor_module.train()
        corrects_pre_sim = 0
        corrects_pre_rumor = 0
        loss_task1_total = 0
        loss_task2_total = 0
        loss_total = 0
        sim_count = 0
        rumor_count = 0

        for i, (text, image, label) in tqdm(enumerate(train_loader)):
            batch_size, token_len = text.shape[0], text.shape[1]

            text = text.to(device)
            image = image.to(device)
            label = label.to(device)

            text = text.reshape(batch_size * token_len, -1)
            text = similarity_module.prepare_mlp(text)
            text = text.reshape(batch_size, token_len, -1).clone().detach()

            fixed_text, matched_image, unmatched_image = prepare_task2_data(text, image, label)
            fixed_text.to(device)
            matched_image.to(device)
            unmatched_image.to(device)

            text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(fixed_text,
                                                                                               matched_image)
            text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(fixed_text,
                                                                                                     unmatched_image)
            similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)
            similarity_label_0 = torch.cat(
                [torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])], dim=0).to(
                device)
            similarity_label_1 = torch.cat(
                [torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])],
                dim=0).to(device)

            text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
            image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
            loss_similarity = loss_f_sim(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)

            optim_task1.zero_grad()
            loss_similarity.backward()
            optim_task1.step()
            # task1中预测正确的个数
            corrects_pre_sim += similarity_pred.eq(similarity_label_0).sum().item()

            # 再训练task2
            text_aligned, image_aligned, _ = similarity_module(text, image)
            pre_rumor = rumor_module(text, image, text_aligned, image_aligned)
            loss_rumor = loss_f_rumor(pre_rumor, label)

            optim_task2.zero_grad()
            loss_rumor.backward()
            optim_task2.step()
            # task2预测正确的个数
            pre_label_rumor = pre_rumor.argmax(1)
            corrects_pre_rumor += pre_label_rumor.eq(label.view_as(pre_label_rumor)).sum().item()

            # 模型的总loss
            loss_task1_total += loss_similarity.item() * (2 * fixed_text.shape[0])
            loss_task2_total += loss_rumor.item() * text.shape[0]
            sim_count += (2 * fixed_text.shape[0] * 2)
            rumor_count += text.shape[0]

            # if (i + 1) % 20 == 0:
            #     print('\n@ Iter-{}: \n- loss_similarity = {}\n- loss_rumor = {}'.format(i, loss_task1_total / sim_count, loss_task2_total / rumor_count))

            if (i + 1) % 10 == 0:
                test_sim, test_rumor, loss_sim_test, loss_rumor_test, conf_sim, conf_rumor = test(
                    similarity_module, rumor_module, test_loader)
                similarity_module.train()
                rumor_module.train()

                if best_acc < test_rumor:
                    best_acc = test_rumor
                    print("- test_rumor = ", test_rumor)
                    print("- conf_rumor = \n", conf_rumor)

        loss_sim_train = loss_task1_total / sim_count
        loss_rumor_train = loss_task2_total / rumor_count

        acc_sim_train = corrects_pre_sim / sim_count
        acc_rumor_train = corrects_pre_rumor / rumor_count

        acc_sim_test, acc_rumor_test, loss_sim_test, loss_rumor_test, conf_sim, conf_rumor = test(similarity_module,
                                                                                                  rumor_module,
                                                                                                  test_loader)
        # task1的输出
        print('-----------similarity modeling------------')
        print(
            "EPOCH = %d || acc_sim_train = %.3f || acc_sim_test = %.3f || loss_sim_train = %.3f || loss_sim_test = %.3f" %
            (epoch + 1, acc_sim_train, acc_sim_test, loss_sim_train, loss_sim_test))
        print('-----------rumor detection----------------')
        print(
            "EPOCH = %d || acc_rumor_train = %.3f || acc_rumor_test = %.3f ||  best_acc = %.3f || loss_rumor_train = %.3f || loss_rumor_test = %.3f" %
            (epoch + 1, acc_rumor_train, acc_rumor_test, best_acc, loss_rumor_train, loss_rumor_test))
        print('-----------corre_confusion_matrix---------')
        print(conf_sim)
        print('-----------rumor_confusion_matrix---------')
        print(conf_rumor)

    # torch.save(best_model_corre, "./result/best_model_corre_" +
    #            str(best_acc)[0:5] + ".pth")
    # torch.save(best_model_rumor, "./result/best_model_rumor_" +
    #            str(best_acc)[0:5] + ".pth")


def test(similarity_module, rumor_module, test_loader):
    similarity_module.eval()
    rumor_module.eval()

    device = torch.device('cuda:1')
    loss_f_rumor = torch.nn.CrossEntropyLoss()
    loss_f_sim = torch.nn.CosineEmbeddingLoss()

    sim_count = 0
    rumor_count = 0
    loss_task1_total = 0
    loss_task2_total = 0
    rumor_label_all = []
    sim_label_all = []
    sim_pre_label_all = []
    rumor_pre_label_all = []
    batch_size = 4

    with torch.no_grad():
        for i, (text, image, label) in enumerate(test_loader):
            batch_size = text.shape[0]

            text = text.to(device)
            image = image.to(device)
            label = label.to(device)
            #  -----------------
            fixed_text, matched_image, unmatched_image = prepare_task2_data(text, image, label)
            fixed_text.to(device)
            matched_image.to(device)
            unmatched_image.to(device)

            text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(fixed_text,
                                                                                               matched_image)
            text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(fixed_text,
                                                                                                     unmatched_image)
            similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)
            similarity_label_0 = torch.cat(
                [torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])], dim=0).to(
                device)
            similarity_label_1 = torch.cat(
                [torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])],
                dim=0).to(device)

            text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
            image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
            loss_similarity = loss_f_sim(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)

            text_aligned, image_aligned, _ = similarity_module(text, image)
            pre_rumor = rumor_module(text, image, text_aligned, image_aligned)
            loss_rumor = loss_f_rumor(pre_rumor, label)
            # task2预测正确的个数
            pre_label_rumor = pre_rumor.argmax(1)

            # total
            loss_task1_total += loss_similarity.item() * (2 * fixed_text.shape[0])
            loss_task2_total += loss_rumor.item() * text.shape[0]

            sim_count += (fixed_text.shape[0] * 2)
            rumor_count += text.shape[0]

            #  ========== predict_label==============
            sim_pre_label_all.append(similarity_pred.detach().cpu().numpy())
            rumor_pre_label_all.append(pre_label_rumor.detach().cpu().numpy())
            sim_label_all.append(similarity_label_0.detach().cpu().numpy())
            rumor_label_all.append(label.detach().cpu().numpy())

        # ---------计算test process全部的loss，acc， confusion matrices-------
        loss_sim_test = loss_task1_total / sim_count
        loss_rumor_test = loss_task2_total / rumor_count

        sim_pre_label_all = np.concatenate(sim_pre_label_all, 0)
        rumor_pre_label_all = np.concatenate(rumor_pre_label_all, 0)
        sim_label_all = np.concatenate(sim_label_all, 0)
        rumor_label_all = np.concatenate(rumor_label_all, 0)

        acc_sim_test = accuracy_score(sim_pre_label_all, sim_label_all)
        acc_rumor_test = accuracy_score(rumor_pre_label_all, rumor_label_all)
        conf_sim = confusion_matrix(sim_pre_label_all, sim_label_all)
        conf_rumor = confusion_matrix(rumor_pre_label_all, rumor_label_all)

    return acc_sim_test, acc_rumor_test, loss_sim_test, loss_rumor_test, conf_sim, conf_rumor


if __name__ == "__main__":
    train()
