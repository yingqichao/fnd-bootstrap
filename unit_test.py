import torch
from CNN_architectures.pytorch_inceptionet import GoogLeNet
# from models.UAMFDv2_Net import UAMFD_Net

optim_params = []
model = GoogLeNet()
for k, v in model.named_parameters():
    print(k)
    # print(v)
    if v.requires_grad:
        optim_params.append(v)