## Bootstrapping Your Own Representations for Fake News Detection

This repo is built upon ["Masked Autoencoders: A PyTorch Implementation"](https://github.com/facebookresearch/mae)

* Data Preparation. You need to prepare the data using the scripts in ./data_prepare. We only support Weibo/Weibo-21/GossipCop so far, and the data should be downloaded exactly from the following sources. Weibo/Weibo-21: send email to Dr [Qiong Nan](nanqiong19z<nanqiong19z@ict.ac.cn). GossipCop: send email to the original authors of GossipCop (say sorry to Dr Singhal for my previous wrong direction and the caused confusion and borthering). They will kindly help (according to our experience)

* After you process the data, run the .sh scripts for training or testing.

* We alternatively provide an alternative in the network design in ./models/UAMFDv2_Net.py, where the differences are trivial: 1) we replaced ELU with SimpleGate where the tensors are split into two halves and the second half is used for reweighing the first half, which also ensures non-linearlity. 2) We use AdaIn to control the mean and std of the refined representations, where the original reweighing MLPs are therefore replaced. Note that if you wish to exactly implement the network design reported in the paper, use UAMFD_Net instead of UAMFDv2_Net, though the latter will be slightly even better according to our later tests.

### Pre-training

The pre-training models of MAE can be downloaded from ["Masked Autoencoders: A PyTorch Implementation"](https://github.com/facebookresearch/mae).

Because of the restriction on upload size, we are unable to upload pretrained models and the processed data. We will further open-source them on GitHub after the anonymous reviewing process.

### License

We have been granted permisson to use Weibo/Weibo-21/GossipCop datasets for academic studies only.
