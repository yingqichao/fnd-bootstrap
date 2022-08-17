import torch
import timm
import numpy as np
import torchvision
from PIL import Image
# vit_large_patch16
import cv2
# model_resnet = timm.list_models('*resnet*')
# model = timm.create_model('vit_large_patch16')
# # print(len(model_resnet), model_resnet[:3])
# print(model)
import models_mae
import timm.models.layers.helpers

with torch.no_grad():
    path = r'./target.jpg'
    img_GT = cv2.imread(path, cv2.IMREAD_COLOR)

    img_GT = img_GT.astype(np.float32) / 255.
    if img_GT.ndim == 2:
        img_GT = np.expand_dims(img_GT, axis=2)
    # some images have 4 channels
    if img_GT.shape[2] > 3:
        img_GT = img_GT[:, :, :3]

    ###### directly resize instead of crop
    img_GT = cv2.resize(np.copy(img_GT), (224, 224),
                        interpolation=cv2.INTER_LINEAR)

    orig_height, orig_width, _ = img_GT.shape
    H, W, _ = img_GT.shape

    # BGR to RGB, HWC to CHW, numpy to tensor
    if img_GT.shape[2] == 3:
        img_GT = img_GT[:, :, [2, 1, 0]]

    img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
    img_GT = img_GT.unsqueeze(0).cuda()
    model = models_mae.__dict__["mae_vit_large_patch16"](norm_pix_loss=False)
    model.cuda()
    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    model.eval()
    checkpoint = torch.load('./mae_pretrain_vit_large.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'],strict=False)
    x = model.forward_ying(img_GT)
    print(x.shape) # torch.Size([1, 197, 1024])
    y = torch.mean(x, 1)
    print(y.shape)

