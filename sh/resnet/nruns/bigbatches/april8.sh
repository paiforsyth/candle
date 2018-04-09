#!/bin/bash
PRUNE_TARG_DECIMAL=30
RUNTYPE=basic_prune
RUNNAME=resnet164_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=07_April_2018_Saturday_13_31_11_endofcycle_checkpoint_0 --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=18 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=54 --squeezenet_pool_interval=18 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=0   --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001  --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context     --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle --fire_skip_mode=zero_pad  --enable_pruning  --prune_target_frac=0.$PRUNE_TARG_DECIMAL --prune_warmup_epochs=0   --autocalc_prune_unit --log_to_file --log_file_name=log_for_$RUNNAME --cuda # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   


PRUNE_TARG_DECIMAL=40
RUNTYPE=basic_prune
RUNNAME=resnet164_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=07_April_2018_Saturday_13_31_11_endofcycle_checkpoint_0 --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=18 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=54 --squeezenet_pool_interval=18 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=0   --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001  --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context     --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle --fire_skip_mode=zero_pad  --enable_pruning  --prune_target_frac=0.$PRUNE_TARG_DECIMAL --prune_warmup_epochs=0   --autocalc_prune_unit --log_to_file --log_file_name=log_for_$RUNNAME --cuda # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   


PRUNE_TARG_DECIMAL=70
RUNTYPE=basic_prune
RUNNAME=resnet164_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=07_April_2018_Saturday_13_31_11_endofcycle_checkpoint_0 --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=18 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=54 --squeezenet_pool_interval=18 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=0   --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001  --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context     --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle --fire_skip_mode=zero_pad  --enable_pruning  --prune_target_frac=0.$PRUNE_TARG_DECIMAL --prune_warmup_epochs=0   --autocalc_prune_unit --log_to_file --log_file_name=log_for_$RUNNAME --cuda # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   


PRUNE_TARG_DECIMAL=80
RUNTYPE=basic_prune
RUNNAME=resnet164_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=07_April_2018_Saturday_13_31_11_endofcycle_checkpoint_0 --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=18 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=54 --squeezenet_pool_interval=18 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=0   --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001  --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context     --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle --fire_skip_mode=zero_pad  --enable_pruning  --prune_target_frac=0.$PRUNE_TARG_DECIMAL --prune_warmup_epochs=0   --autocalc_prune_unit --log_to_file --log_file_name=log_for_$RUNNAME --cuda # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   


RUNTYPE=l0_reg
NETTYPE=wrn_28_10
RUNNAME=${NETTYPE}_${RUNTYPE}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNAME --dataset_for_classification=cifar10 --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=zag_fire --squeezenet_zag_fire_dropout=0  --squeezenet_pool_interval_mode=multiply  --squeezenet_base=160 --squeezenet_freq=4   --squeezenet_num_fires=12 --squeezenet_pool_interval=4 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=0   --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=300 --optimizer=sgd --sgd_momentum=0.9   --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=3   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1  --proxy_context_type=l0reg_context --enable_l0reg --l0reg_lambda=0.0000001 --use_all_params --enable_l2reg_stochastic --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle --fire_skip_mode=zero_pad --grad_norm_clip=50 --cuda     --log_to_file --log_file_name=log_for_$RUNNAME --cuda # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   


