CUDA_VISIBLE_DEVICES=1 python ./UAMFD.py -train_dataset gossip \
                                        -test_dataset gossip \
                                        -batch_size 24 \
                                        -epochs 50 \
                                        -checkpoint /home/groupshare/CIKM_ying_output//gossip/29_810_89.pkl \
                                        -network_arch UAMFD \
                                        -val 1
