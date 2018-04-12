#!/bin/bash
PRUNE_TARG_DECIMAL=90
RUNTYPE=slimming
RUNNAME=resnet164_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_fixedmaxpool
FINAL_L_EPOCH=113
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=cifar10 --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=18 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=54 --squeezenet_pool_interval=18 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=0 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=200 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=2   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1  --proxy_context_type=l1reg_context_slimming  --enable_l1reg  --squeezenet_bypass_first_last --prune_layer_mode=global --save_every_epoch --disable_l1_reg_after_epoch=${FINAL_L_EPOCH}  --use_no_grad --report_unpruned  --count_multiplies_every_cycle  --enable_pruning --prune_target_frac=0.$PRUNE_TARG_DECIMAL --autocalc_prune_unit  --prune_warmup_epochs=102  --fire_skip_mode=zero_pad --log_to_file --log_file_name=log_for_$RUNNAME  --cuda 



