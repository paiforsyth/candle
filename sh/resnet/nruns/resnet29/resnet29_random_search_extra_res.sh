#!/bin/bash
for PRUNE_TARG_DECIMAL in  97 95 92 90 87 85 82 35 25 15 5; do

RUNTYPE=random
NETNAME=resnet29
PRUNE_STRAT=random
RUNNAME=${NETNAME}_${RUNTYPE}_${PRUNE_STRAT}_${PRUNE_TARG_DECIMAL}_fixedmaxpool
BLOCK_OFFSET=1
python -m examples.lab --group_prune_strategy=$PRUNE_STRAT --resume_mode=standard --res_file=09_April_2018_Monday_02_17_27_endofcycle_checkpoint_0 --save_prefix=$RUNNAME --dataset_for_classification=cifar10 --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3 --squeezenet_freq_offset=0 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=$BLOCK_OFFSET   --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001  --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context     --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle --fire_skip_mode=zero_pad  --enable_pruning  --prune_target_frac=0.$PRUNE_TARG_DECIMAL --prune_warmup_epochs=0   --autocalc_prune_unit --cuda --log_to_file --log_file_name=log_for_$RUNNAME      --report_test_error_at_end   --report_test_error_before_start     # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   

done
