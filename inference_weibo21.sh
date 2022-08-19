CUDA_VISIBLE_DEVICES=0 python ./UAMFD.py -train_dataset weibo \
                                        -test_dataset weibo \
                                        -batch_size 24 \
                                        -epochs 50 \
                                        -checkpoint /groupshare/CIKM_ying_output//weibo/1_818_91.pkl \
                                        -network_arch UAMFDv2 \
                                        -val 1 \
                                        -get_MLP_score 0 \
                                        -not_on_12 1

#-network_arch UAMFD \
#/home/groupshare/CIKM_ying_output//weibo/46_810_91.pkl