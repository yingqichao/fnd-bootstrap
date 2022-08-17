CUDA_VISIBLE_DEVICES=1 python ./UAMFD.py -train_dataset weibo \
                                        -test_dataset weibo \
                                        -batch_size 16 \
                                        -epochs 50 \
                                        -val 0 \
                                        -is_sample_positive 1.0 \
                                        -duplicate_fake_times 0 \
                                        -network_arch UAMFDv2 \
                                        -is_filter 0 \
                                        -not_on_12 1