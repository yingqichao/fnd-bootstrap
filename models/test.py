from positional_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
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
from timm.models.vision_transformer import Block

class SimpleGate(nn.Module):
    def forward(self,x):
        x1,x2 = x.chunk(2,dim=1)
        return x1*x2

class TokenAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class UAMFD_Net(nn.Module):
    def __init__(self, batch_size=1, dataset='weibo', text_token_len=197, image_token_len=197, is_use_bce=True):
        # NOTE: NOW WE ONLY SUPPORT BASE MODEL!
        self.batch_size = batch_size
        self.text_token_len, self.image_token_len = text_token_len, image_token_len
        model_size = 'base'
        self.model_size = model_size
        self.dataset = dataset
        self.LOW_BATCH_SIZE_AND_LR = ['Twitter', 'politi']

        self.unified_dim, self.text_dim = 768, 768
        self.is_use_bce = is_use_bce
        out_dim = 1 if self.is_use_bce else 2
        self.num_expert = 3  # 2
        self.depth = 1  # 2
        super(UAMFD_Net, self).__init__()
        # IMAGE: MAE
        # if self.dataset in self.LOW_BATCH_SIZE_AND_LR:
        #     self.image_model = Block(dim=self.unified_dim, num_heads=16)
        # else:
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        # for param in self.image_model.parameters():
        #     param.requires_grad = False

        # image_model_finetune = []
        # self.finetune_depth = 1
        # for j in range(self.finetune_depth):
        #     image_model_finetune.append(Block(dim=self.unified_dim, num_heads=16))  # note: need to output model[:,0]
        # self.image_model_finetune = nn.ModuleList(image_model_finetune)

        # TEXT: BERT OR PRETRAINED FROM WWW
        english_lists = ['gossip', 'Twitter', 'politi']
        model_name = 'bert-base-chinese' if self.dataset not in english_lists else 'bert-base-uncased'
        print("BERT: using {}".format(model_name))
        # if self.dataset in self.LOW_BATCH_SIZE_AND_LR:
        #     self.text_model = Block(dim=self.unified_dim, num_heads=16)
        # else:
        self.text_model = BertModel.from_pretrained(model_name)
        # for param in self.text_model.parameters():
        #     param.requires_grad = False
        # text_model_finetune = []
        # for j in range(self.finetune_depth):
        #     text_model_finetune.append(Block(dim=self.unified_dim, num_heads=16))  # note: need to output model[:,0]
        # self.text_model_finetune = nn.ModuleList(text_model_finetune)

        self.text_attention = TokenAttention(self.unified_dim)

        # IMAGE: RESNET-50
        # self.vgg_net = torchvision.models.resnet50(pretrained=False)
        # self.vgg_net.fc = nn.Sequential(
        #     nn.Linear(2048, self.unified_dim)
        # )
        # from CNN_architectures.pytorch_resnet import ResNet50
        # self.vgg_net = ResNet50(img_channel=3, num_classes=self.unified_dim, use_SRM=True)
        from CNN_architectures.pytorch_inceptionet import GoogLeNet
        self.vgg_net = GoogLeNet(num_classes=self.unified_dim, use_SRM=False)

        # self.vgg_net = self.vgg_net
        self.image_attention = TokenAttention(self.unified_dim)

        self.mm_attention = TokenAttention(self.unified_dim)
        # GATE, EXPERTS
        # feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64} # 64*5 note there are 5 kernels and 5 experts!
        image_expert_list, text_expert_list, mm_expert_list = [], [], []

        for i in range(self.num_expert):
            image_expert = []
            for j in range(self.depth):
                image_expert.append(Block(dim=self.unified_dim, num_heads=16))  # note: need to output model[:,0]

            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)

        for i in range(self.num_expert):
            text_expert = []
            mm_expert = []
            for j in range(self.depth):
                text_expert.append(Block(dim=self.unified_dim, num_heads=16))
                mm_expert.append(Block(dim=self.unified_dim, num_heads=16))

            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
            mm_expert = nn.ModuleList(mm_expert)
            mm_expert_list.append(mm_expert)

        self.image_experts = nn.ModuleList(image_expert_list)
        self.text_experts = nn.ModuleList(text_expert_list)
        self.mm_experts = nn.ModuleList(mm_expert_list)
        # self.out_unified_dim = 320
        self.image_gate_mae = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                            nn.ELU(inplace=True),
                                            nn.BatchNorm1d(self.unified_dim),
                                            nn.Linear(self.unified_dim, self.num_expert),
                                            # nn.Dropout(0.1),
                                            # nn.Softmax(dim=1)
                                            )
        # self.image_gate_vgg = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
        #                                 nn.LayerNorm(self.unified_dim),
        #                                 SimpleGate(), # nn.SiLU(),
        #                                 nn.Dropout(0.2),
        #                                 nn.Linear(self.unified_dim, 1),
        #                                 nn.Softmax(dim=1)
        #                                 )

        self.text_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                       nn.ELU(inplace=True),
                                       nn.BatchNorm1d(self.unified_dim),
                                       nn.Linear(self.unified_dim, self.num_expert),
                                       # nn.Dropout(0.1),
                                       # nn.Softmax(dim=1)
                                       )
        self.mm_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                     nn.ELU(inplace=True),
                                     nn.BatchNorm1d(self.unified_dim),
                                     nn.Linear(self.unified_dim, self.num_expert),
                                     # nn.Dropout(0.1),
                                     # nn.Softmax(dim=1)
                                     )

        self.image_gate_mae_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                              nn.ELU(inplace=True),
                                              nn.BatchNorm1d(self.unified_dim),
                                              nn.Linear(self.unified_dim, self.num_expert),
                                              # nn.Dropout(0.1),
                                              # nn.Softmax(dim=1)
                                              )

        self.text_gate_1 = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                         nn.ELU(inplace=True),
                                         nn.BatchNorm1d(self.unified_dim),
                                         nn.Linear(self.unified_dim, self.num_expert),
                                         # nn.Dropout(0.1),
                                         # nn.Softmax(dim=1)
                                         )

        # self.mm_gate_1 = nn.Sequential(nn.Linear(self.unified_dim + self.unified_dim, self.unified_dim),
        #                                SimpleGate(),
        #                                nn.BatchNorm1d(int(self.unified_dim / 2)),
        #                                nn.Linear(int(self.unified_dim/2), self.num_expert),
        #                                # nn.Dropout(0.1),
        #                                # nn.Softmax(dim=1)
        #                                )

        ## MAIN TASK GATES
        self.final_attention = TokenAttention(self.unified_dim)

        # self.mm_SE_network_main_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                              SimpleGate(),
        #                                              nn.BatchNorm1d(128),
        #                                              # nn.Dropout(0.2),
        #                                              )
        # self.text_SE_network_main_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                                SimpleGate(),
        #                                                nn.BatchNorm1d(128),
        #                                                # nn.Dropout(0.2),
        #                                                )
        # self.image_SE_network_main_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                                 SimpleGate(),
        #                                                 nn.BatchNorm1d(128),
        #                                                 # nn.Dropout(0.2),
        #                                                 )
        # self.vgg_SE_network_main_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                               SimpleGate(),
        #                                               nn.BatchNorm1d(128),
        #                                               # nn.Dropout(0.2),
        #                                               )
        self.fusion_SE_network_main_task = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                                         nn.ELU(inplace=True),
                                                         nn.BatchNorm1d(self.unified_dim),
                                                         nn.Linear(self.unified_dim, self.num_expert),
                                                         # nn.Softmax(dim=1)
                                                         )
        ## AUXILIARY TASK GATES
        # self.mm_SE_network_aux_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                              nn.LayerNorm(256),
        #                                              SimpleGate(), # nn.SiLU(),
        #                                              nn.Dropout(0.2),
        #                                              )
        # self.text_SE_network_aux_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                                nn.LayerNorm(256),
        #                                                SimpleGate(), # nn.SiLU(),
        #                                                nn.Dropout(0.2),
        #                                                )
        # self.image_SE_network_aux_task = nn.Sequential(nn.Linear(self.unified_dim, 256),
        #                                                 nn.LayerNorm(256),
        #                                                 SimpleGate(), # nn.SiLU(),
        #                                                 nn.Dropout(0.2),
        #                                                 )
        # self.fusion_SE_network_aux_task = nn.Sequential(nn.Linear(3 * 256, 256),
        #                                                  nn.LayerNorm(256),
        #                                                  SimpleGate(), # nn.SiLU(),
        #                                                  nn.Dropout(0.2),
        #                                                  nn.Linear(256, self.num_expert),
        #                                                  nn.Softmax(dim=1)
        #                                                  )

        # self.irre_dim = self.unified_dim
        # self.irre_feature = None
        # self.irr_MLP = nn.Linear(self.unified_dim, self.unified_dim)
        # self.irrelevant_tensor = nn.Parameter(torch.rand((1,self.irre_dim)))
        # self.irrelevant_token = nn.Parameter(torch.randn(self.unified_dim),requires_grad=True) #torch.zeros_like(shared_mm_feature)
        ## POSITIONAL ENCODING FOR MM FEATURE
        p_1d_mm = PositionalEncoding1D(self.unified_dim)
        x_mm = torch.rand(self.batch_size, self.image_token_len + self.text_token_len, self.unified_dim)
        self.positional_mm = p_1d_mm(x_mm)
        p_1d_image = PositionalEncoding1D(self.unified_dim)
        x_image = torch.rand(self.batch_size, self.image_token_len, self.unified_dim)
        self.positional_image = p_1d_image(x_image)
        p_1d_text = PositionalEncoding1D(self.unified_dim)
        x_text = torch.rand(self.batch_size, self.text_token_len, self.unified_dim)
        self.positional_text = p_1d_text(x_text)
        ## POSITIONAL ENCODING FOR MULTIMODAL
        ## 5 IS BECAUSE THE MODALS ARE IMAGE TEXT MM IRRELEVANT AND VGG
        p_1d = PositionalEncoding1D(self.unified_dim)
        x = torch.rand(self.batch_size, 5, self.unified_dim)
        self.positional_modal_representation = p_1d(x)
        # self.positional_imagefeature = pos[:, 0, :].squeeze()
        # self.positional_textfeature = pos[:, 1, :].squeeze()
        # self.positional_mmfeature = pos[:, 2, :].squeeze()
        # self.irrelevant_tensor = pos[:,3,:].squeeze()
        # print(pos)
        # print(pos[:, 25, :])

        # self.mm_SE_network_classifier = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
        #                              nn.LayerNorm(self.unified_dim),
        #                              SimpleGate(), # nn.SiLU(),
        #                              nn.Dropout(0.2),
        #                              nn.Linear(self.unified_dim, 2)
        #                                    )

        # CLASSIFICATION HEAD
        self.mix_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.mix_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.text_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.text_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.image_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.image_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.vgg_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.vgg_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.aux_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.aux_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        final_fusing_expert = []
        for i in range(self.num_expert):
            fusing_expert = []
            for j in range(self.depth):
                fusing_expert.append(Block(dim=self.unified_dim, num_heads=16))
            fusing_expert = nn.ModuleList(fusing_expert)
            final_fusing_expert.append(fusing_expert)

        self.final_fusing_experts = nn.ModuleList(final_fusing_expert)

        self.mm_score = None

        # self.final_fusing_experts = Block(dim=self.unified_dim, num_heads=16)

    def get_pretrain_features(self, input_ids, attention_mask, token_type_ids, image, no_ambiguity, category=None,
                              calc_ambiguity=False):
        image_feature = self.image_model.forward_ying(image)
        text_feature = self.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)[0]

        return image_feature, text_feature

    def forward(self, image=torch.randn((1,3,224,224)),
                # input_ids=torch.ones((1,197),dtype=torch.int32),
                # attention_mask=torch.randn((1, 197),dtype=torch.int32),
                # token_type_ids=torch.randn((1, 197),dtype=torch.int32),
                image_feature=torch.randn((1,197,768)),
                text_feature=torch.randn((1,197,768)),
                calc_ambiguity=False,
                return_features=False):


        image_feature = self.image_model.forward_ying(image)

        vgg_feature = self.vgg_net(image)  # 64, 768
        # print(vgg_feature.shape)

        # text_feature = self.text_model(input_ids=input_ids,
        #                                attention_mask=attention_mask,
        #                                token_type_ids=token_type_ids)[0]

        text_atn_feature, _ = self.text_attention(text_feature)


        image_atn_feature, _ = self.image_attention(image_feature)

        mm_atn_feature, _ = self.mm_attention(torch.cat((image_feature, text_feature), dim=1))


        gate_image_feature = self.image_gate_mae(image_atn_feature)
        gate_text_feature = self.text_gate(text_atn_feature)  # 64 320
        gate_mm_feature = self.mm_gate(mm_atn_feature)


        shared_image_feature, shared_image_feature_1 = 0, 0

        for i in range(self.num_expert):
            image_expert = self.image_experts[i]
            tmp_image_feature = image_feature
            for j in range(self.depth):
                tmp_image_feature = image_expert[j](tmp_image_feature + self.positional_image)
            shared_image_feature += (tmp_image_feature * gate_image_feature[:, i].unsqueeze(1).unsqueeze(1))

        shared_image_feature = shared_image_feature[:, 0]


        shared_text_feature, shared_text_feature_1 = 0, 0
        for i in range(self.num_expert):
            text_expert = self.text_experts[i]
            tmp_text_feature = text_feature
            for j in range(self.depth):
                tmp_text_feature = text_expert[j](tmp_text_feature + self.positional_text)  # text_feature: 64, 170, 768
            shared_text_feature += (tmp_text_feature * gate_text_feature[:, i].unsqueeze(1).unsqueeze(1))
        shared_text_feature = shared_text_feature[:, 0]

        mm_feature = torch.cat((image_feature, text_feature), dim=1)
        shared_mm_feature = 0
        for i in range(self.num_expert):
            mm_expert = self.mm_experts[i]
            tmp_mm_feature = mm_feature
            for j in range(self.depth):
                tmp_mm_feature = mm_expert[j](tmp_mm_feature + self.positional_mm)
            shared_mm_feature += (tmp_mm_feature * gate_mm_feature[:, i].unsqueeze(1).unsqueeze(1))
        shared_mm_feature = shared_mm_feature[:, 0]

        shared_mm_feature_lite = self.aux_trim(shared_mm_feature)
        aux_output = self.aux_classifier(shared_mm_feature_lite)  # final_feature_aux_task


        vgg_feature_lite = self.vgg_trim(vgg_feature)
        shared_image_feature_lite = self.image_trim(shared_image_feature)
        shared_text_feature_lite = self.text_trim(shared_text_feature)

        vgg_only_output = self.vgg_alone_classifier(vgg_feature_lite)
        image_only_output = self.image_alone_classifier(shared_image_feature_lite)
        text_only_output = self.text_alone_classifier(shared_text_feature_lite)

        aux_atn_score = 1 - torch.sigmoid(aux_output).clone().detach()  # torch.abs((torch.sigmoid(aux_output).clone().detach()-0.5)*2)
        image_atn_score = torch.abs((torch.sigmoid(image_only_output).clone().detach() - 0.5) * 2)
        text_atn_score = torch.abs((torch.sigmoid(text_only_output).clone().detach() - 0.5) * 2)
        vgg_atn_score = torch.abs((torch.sigmoid(vgg_only_output).clone().detach() - 0.5) * 2)
        irre_atn_score = 1-aux_atn_score

        # print(image_atn_score.shape)

        shared_image_feature = shared_image_feature * (image_atn_score)
        shared_text_feature = shared_text_feature * (text_atn_score)
        shared_mm_feature = shared_mm_feature * (aux_atn_score)
        vgg_feature = vgg_feature * (vgg_atn_score)

        # print(shared_image_feature.shape)

        irr_score = torch.ones_like(shared_mm_feature)
        irrelevant_token = irr_score * (irre_atn_score)
        concat_feature_main_biased = torch.stack((shared_image_feature,
                                                  shared_text_feature,
                                                  shared_mm_feature,
                                                  vgg_feature,
                                                  irrelevant_token
                                                  ), dim=1)

        fusion_tempfeat_main_task, _ = self.final_attention(concat_feature_main_biased)
        gate_main_task = self.fusion_SE_network_main_task(fusion_tempfeat_main_task)

        final_feature_main_task = 0
        for i in range(self.num_expert):
            fusing_expert = self.final_fusing_experts[i]
            tmp_fusion_feature = concat_feature_main_biased
            for j in range(self.depth):
                tmp_fusion_feature = fusing_expert[j](tmp_fusion_feature + self.positional_modal_representation)
            tmp_fusion_feature = tmp_fusion_feature[:, 0]
            final_feature_main_task += (tmp_fusion_feature * gate_main_task[:, i].unsqueeze(1))

        final_feature_main_task_lite = self.mix_trim(final_feature_main_task)
        mix_output = self.mix_classifier(final_feature_main_task_lite)

        return mix_output, image_only_output, text_only_output, vgg_only_output, aux_output, \
               torch.mean(aux_output), \
               (final_feature_main_task_lite, shared_image_feature_lite, shared_text_feature_lite, vgg_feature_lite, shared_mm_feature_lite)


if __name__ == '__main__':
    from thop import profile
    model = UAMFD_Net()
    device = torch.device("cpu")
    input1 = torch.randn(1,197,768)
    input2 = torch.randn(1, 197, 768)
    flops,params = profile(model,inputs=(input1,input2))
    print(f"FLOPs={str(flops/1e9)}G")
    print(f"params={str(params / 1e6)}M")

    # stat(self.localizer.to(torch.device('cuda:0')), (3, 512, 512))
