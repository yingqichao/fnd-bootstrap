CUDA_VISIBLE_DEVICES=0 python ./UAMFD.py -train_dataset weibo \
                                        -test_dataset weibo \
                                        -batch_size 12 \
                                        -epochs 50 \
                                        -val 0 \
                                        -is_sample_positive 1.0 \
                                        -duplicate_fake_times 0 \
                                        -network_arch UAMFDv2 \
                                        -is_filter 0 \
                                        -checkpoint /groupshare/CIKM_ying_output//weibo/26_818_90.pkl \
                                        -not_on_12 1
#-checkpoint /groupshare/CIKM_ying_output//weibo/18_812_91.pkl \