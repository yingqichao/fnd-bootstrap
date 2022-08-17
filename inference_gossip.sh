CUDA_VISIBLE_DEVICES=1 python ./UAMFD.py -train_dataset gossip \
                                        -test_dataset gossip \
                                        -batch_size 24 \
                                        -epochs 50 \
                                        -checkpoint /groupshare/CIKM_ying_output//gossip/17_815_89.pkl \
                                        -network_arch UAMFDv2 \
                                        -is_filter 1 \
                                        -val 1 \
                                        -get_MLP_score 1 \
                                        -not_on_12 1
