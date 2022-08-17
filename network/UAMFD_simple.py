class UAMFD_Net(nn.Module):
    def __init__(self, dataset='weibo',hidden_dim=256):
        self.dataset = dataset
        self.num_expert = 5
        self.img_dim, self.txt_dim = 1024,768
        self.hidden_size = hidden_dim
        super(UAMFD_Net, self).__init__()
        self.image_model = models_mae.__dict__["mae_vit_large_patch16"](norm_pix_loss=False)
        self.image_model.cuda()
        # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        checkpoint = torch.load('./mae_pretrain_vit_large.pth', map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict = False)
        for param in self.image_model.parameters():
            param.requires_grad = False

        # TEXT model
        model_name = 'bert-base-chinese' if self.dataset !='Twitter' and self.dataset != 'Fake' else 'bert-base-uncased'
        self.text_model = BertModel.from_pretrained(model_name).cuda()
        self.netG = torchvision.models.inception_v3(pretrained=True)
        self.netG.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Dropout(0.25),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.netG = self.netG.cuda()
        for param in self.text_model.parameters():
            param.requires_grad = False
        # self.text_fc1 = nn.Sequential(
        #     nn.Linear(768, hidden_dim),
        #     nn.GELU()
        # )

        # IMAGE
        # hidden_size = args.hidden_dim
        # vgg_19 = torchvision.models.vgg19(pretrained=True)
        # visual model

        # self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        # self.image_fc2 = nn.Linear(512, self.hidden_size)
        # self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        # self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        self.text_attention = MaskAttention(768)
        self.image_attention = MaskAttention(1024)
        self.image_gate = nn.Sequential(nn.Linear(self.img_dim, 384),
                                    nn.Dropout(0.25),
                                    nn.BatchNorm1d(384),
                                  nn.GELU(),
                                  nn.Linear(384, self.num_expert),
                                  nn.Softmax(dim=1))

        self.txt_gate = nn.Sequential(nn.Linear(self.txt_dim, 384),
                                      nn.Dropout(0.25),
                                      nn.BatchNorm1d(384),
                                  nn.GELU(),
                                  nn.Linear(384, self.num_expert),
                                  nn.Softmax(dim=1))

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64} # 64*5 note there are 5 kernels and 5 experts!
        img_expert, txt_expert = [], []
        for i in range(self.num_expert):
            img_expert.append(cnn_extractor(feature_kernel, self.img_dim))
            txt_expert.append(cnn_extractor(feature_kernel, self.txt_dim))
        self.img_experts = nn.ModuleList(img_expert)
        self.txt_experts = nn.ModuleList(txt_expert)

        self.out_txt_dim, self.out_img_dim = 320, 320
        # self.text_alone_attn = Block(dim=self.out_txt_dim, num_heads=16)
        # self.image_alone_attn = Block(dim=self.out_img_dim, num_heads=16)
        # self.mix_attn = Block(dim=self.out_txt_dim + self.out_img_dim, num_heads=16)
        self.text_alone_attn = nn.Sequential(
            nn.Linear(self.out_txt_dim,self.out_txt_dim),
            nn.Dropout(0.25),
            nn.BatchNorm1d(self.out_txt_dim),
            nn.GELU(),
        )
        self.image_alone_attn = nn.Sequential(
            nn.Linear(self.out_img_dim, self.out_img_dim),
            nn.Dropout(0.25),
            nn.BatchNorm1d(self.out_img_dim),
            nn.GELU(),
        )
        self.mix_attn = nn.Sequential(
            nn.Linear(self.out_txt_dim + self.out_img_dim, self.out_txt_dim + self.out_img_dim),
            nn.Dropout(0.25),
            nn.BatchNorm1d(self.out_txt_dim + self.out_img_dim),
            nn.GELU(),
        )
        self.img_alone_classifier = nn.Sequential(
            nn.Linear(self.out_img_dim,2)
        )
        self.txt_alone_classifier = nn.Sequential(
            nn.Linear(self.out_txt_dim, 2)
        )
        self.mix_classifier = nn.Sequential(
            nn.Linear(self.out_txt_dim + self.out_img_dim, self.out_img_dim),
            nn.Dropout(0.25),
            nn.BatchNorm1d(self.out_img_dim),
            nn.GELU(),
            nn.Linear(self.out_img_dim, 2),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, image, imageclip=None, textclip=None):

        ### IMAGE #####
        image_feature = self.image_model.forward_ying(image)
        # image_feature = torch.mean(image_feature, 1) # (B,1024)
        ###TEXT###
        text_feature = self.text_model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)[0]
        # 原来的方案：直接取第一个token
        # text_feature = text_feature.last_hidden_state[:, 0]
        # 或者对所有的token取平均
        # text_feature = torch.mean(text_feature, 1) # (B,1024)
        # 现在：跟mdfend靠齐，加上MMoE思想
        # print("text_feature size {}".format(text_feature.shape)) # 64,170,768
        # print("image_feature size {}".format(image_feature.shape)) # 64,197,1024
        txt_atn_feature, _ = self.text_attention(text_feature, attention_mask)
        img_atn_feature, _ = self.image_attention(image_feature, None)
        # print("txt_atn_feature size {}".format(txt_atn_feature.shape))
        # print("img_atn_feature size {}".format(img_atn_feature.shape))
        gate_img_feature = self.image_gate(img_atn_feature)
        gate_txt_feature = self.txt_gate(txt_atn_feature) # 64 320

        shared_img_feature, shared_txt_feature = 0, 0
        for i in range(self.num_expert):
            tmp_image_feature = self.img_experts[i](image_feature)
            tmp_text_feature = self.txt_experts[i](text_feature)
            shared_img_feature += (tmp_image_feature * gate_img_feature[:, i].unsqueeze(1))
            shared_txt_feature += (tmp_text_feature * gate_txt_feature[:, i].unsqueeze(1))

        ### text-only branch
        # print("shared_img_feature {}".format(shared_img_feature.shape))
        # print("shared_txt_feature {}".format(shared_txt_feature.shape))
        text_alone_output = self.txt_alone_classifier(self.text_alone_attn(shared_txt_feature))
        ### image-only branch
        image_alone_output = self.img_alone_classifier(self.image_alone_attn(shared_img_feature))

        ### mixed branch
        concat_feature = torch.cat((shared_img_feature, shared_txt_feature), dim=1)
        mix_output = self.mix_classifier(self.mix_attn(concat_feature))

        return text_alone_output, image_alone_output, mix_output

