#!/bin/bash
RUNTYPE=l0_reg_adam_slow
NETTYPE=resnet29
RUNNAME=${NETTYPE}_${RUNTYPE}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNAME --dataset_for_classification=cifar10 --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1  --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3   --squeezenet_num_fires=9 --squeezenet_sr=0.25 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=1   --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=300 --optimizer=adam   --init_lr=0.00001 --squeezenet_num_layer_chunks=1  --proxy_context_type=l0reg_context --enable_l0reg --l0reg_lambda=0.0000001 --use_all_params  --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle --cifar_random_erase --fire_skip_mode=zero_pad --grad_norm_clip=50   --cuda   --log_to_file --log_file_name=log_for_$RUNNAME  # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   



