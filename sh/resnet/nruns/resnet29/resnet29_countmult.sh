#!/bin/bash
PRUNE_TARG_DECIMAL=90
RUNTYPE=group_context_no_prune
NETNAME=resnet29
RUNNAME=${NETNAME}_${RUNTYPE}__fixedmaxpool
BLOCK_OFFSET=1
python -m examples.lab --save_prefix=$RUNAME --dataset_for_classification=cifar10 --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3 --squeezenet_freq_offset=0 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=$BLOCK_OFFSET   --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001  --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context     --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle --fire_skip_mode=zero_pad  --enable_pruning  --prune_target_frac=0.$PRUNE_TARG_DECIMAL --prune_warmup_epochs=102   --autocalc_prune_unit    --count_multiplies --output_level=debug              # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   



