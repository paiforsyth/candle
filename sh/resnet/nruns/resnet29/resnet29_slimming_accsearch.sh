#!/bin/bash
PRUNE_TARG_DECIMAL=01
RUNTYPE=slimming
NETNAME=resnet29
PRUNE_START_EPOCH=0
L1_LAMBDA=0.00001
REG_LAST_EPOCH=$(($PRUNE_START_EPOCH + 13  ))
RUNNAME=${NETNAME}_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_lambda${L1_LAMBDA}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=08_April_2018_Sunday_14_50_19resnet29_slimming_50_lambda0.00001_fixedmaxpool_endofcycle_checkpoint_0  --disable_l1_reg_after_epoch  --l1_reg_final_epoch=$REG_LAST_EPOCH --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=1 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=l1reg_context_slimming  --enable_l1reg  --l1reg_lambda=$L1_LAMBDA --squeezenet_bypass_first_last --prune_layer_mode=global --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle  --enable_pruning --prune_target_frac=0.$PRUNE_TARG_DECIMAL --autocalc_prune_unit  --prune_warmup_epochs=$PRUNE_START_EPOCH  --fire_skip_mode=zero_pad --log_to_file --log_file_name=log_for_$RUNNAME  --cuda 

PRUNE_TARG_DECIMAL=10
RUNTYPE=slimming
NETNAME=resnet29
PRUNE_START_EPOCH=0
L1_LAMBDA=0.00001
REG_LAST_EPOCH=$(($PRUNE_START_EPOCH + 13  ))
RUNNAME=${NETNAME}_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_lambda${L1_LAMBDA}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=08_April_2018_Sunday_14_50_19resnet29_slimming_50_lambda0.00001_fixedmaxpool_endofcycle_checkpoint_0  --disable_l1_reg_after_epoch  --l1_reg_final_epoch=$REG_LAST_EPOCH --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=1 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=l1reg_context_slimming  --enable_l1reg  --l1reg_lambda=$L1_LAMBDA --squeezenet_bypass_first_last --prune_layer_mode=global --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle  --enable_pruning --prune_target_frac=0.$PRUNE_TARG_DECIMAL --autocalc_prune_unit  --prune_warmup_epochs=$PRUNE_START_EPOCH  --fire_skip_mode=zero_pad --log_to_file --log_file_name=log_for_$RUNNAME  --cuda 

PRUNE_TARG_DECIMAL=20
RUNTYPE=slimming
NETNAME=resnet29
PRUNE_START_EPOCH=0
L1_LAMBDA=0.00001
REG_LAST_EPOCH=$(($PRUNE_START_EPOCH + 13  ))
RUNNAME=${NETNAME}_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_lambda${L1_LAMBDA}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=08_April_2018_Sunday_14_50_19resnet29_slimming_50_lambda0.00001_fixedmaxpool_endofcycle_checkpoint_0  --disable_l1_reg_after_epoch  --l1_reg_final_epoch=$REG_LAST_EPOCH --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=1 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=l1reg_context_slimming  --enable_l1reg  --l1reg_lambda=$L1_LAMBDA --squeezenet_bypass_first_last --prune_layer_mode=global --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle  --enable_pruning --prune_target_frac=0.$PRUNE_TARG_DECIMAL --autocalc_prune_unit  --prune_warmup_epochs=$PRUNE_START_EPOCH  --fire_skip_mode=zero_pad --log_to_file --log_file_name=log_for_$RUNNAME  --cuda 

PRUNE_TARG_DECIMAL=30
RUNTYPE=slimming
NETNAME=resnet29
PRUNE_START_EPOCH=0
L1_LAMBDA=0.00001
REG_LAST_EPOCH=$(($PRUNE_START_EPOCH + 13  ))
RUNNAME=${NETNAME}_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_lambda${L1_LAMBDA}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=08_April_2018_Sunday_14_50_19resnet29_slimming_50_lambda0.00001_fixedmaxpool_endofcycle_checkpoint_0  --disable_l1_reg_after_epoch  --l1_reg_final_epoch=$REG_LAST_EPOCH --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=1 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=l1reg_context_slimming  --enable_l1reg  --l1reg_lambda=$L1_LAMBDA --squeezenet_bypass_first_last --prune_layer_mode=global --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle  --enable_pruning --prune_target_frac=0.$PRUNE_TARG_DECIMAL --autocalc_prune_unit  --prune_warmup_epochs=$PRUNE_START_EPOCH  --fire_skip_mode=zero_pad --log_to_file --log_file_name=log_for_$RUNNAME  --cuda 

PRUNE_TARG_DECIMAL=40
RUNTYPE=slimming
NETNAME=resnet29
PRUNE_START_EPOCH=0
L1_LAMBDA=0.00001
REG_LAST_EPOCH=$(($PRUNE_START_EPOCH + 13  ))
RUNNAME=${NETNAME}_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_lambda${L1_LAMBDA}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=08_April_2018_Sunday_14_50_19resnet29_slimming_50_lambda0.00001_fixedmaxpool_endofcycle_checkpoint_0  --disable_l1_reg_after_epoch  --l1_reg_final_epoch=$REG_LAST_EPOCH --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=1 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=l1reg_context_slimming  --enable_l1reg  --l1reg_lambda=$L1_LAMBDA --squeezenet_bypass_first_last --prune_layer_mode=global --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle  --enable_pruning --prune_target_frac=0.$PRUNE_TARG_DECIMAL --autocalc_prune_unit  --prune_warmup_epochs=$PRUNE_START_EPOCH  --fire_skip_mode=zero_pad --log_to_file --log_file_name=log_for_$RUNNAME  --cuda 

PRUNE_TARG_DECIMAL=60
RUNTYPE=slimming
NETNAME=resnet29
PRUNE_START_EPOCH=0
L1_LAMBDA=0.00001
REG_LAST_EPOCH=$(($PRUNE_START_EPOCH + 13  ))
RUNNAME=${NETNAME}_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_lambda${L1_LAMBDA}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=08_April_2018_Sunday_14_50_19resnet29_slimming_50_lambda0.00001_fixedmaxpool_endofcycle_checkpoint_0  --disable_l1_reg_after_epoch  --l1_reg_final_epoch=$REG_LAST_EPOCH --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=1 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=l1reg_context_slimming  --enable_l1reg  --l1reg_lambda=$L1_LAMBDA --squeezenet_bypass_first_last --prune_layer_mode=global --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle  --enable_pruning --prune_target_frac=0.$PRUNE_TARG_DECIMAL --autocalc_prune_unit  --prune_warmup_epochs=$PRUNE_START_EPOCH  --fire_skip_mode=zero_pad --log_to_file --log_file_name=log_for_$RUNNAME  --cuda 

PRUNE_TARG_DECIMAL=70
RUNTYPE=slimming
NETNAME=resnet29
PRUNE_START_EPOCH=0
L1_LAMBDA=0.00001
REG_LAST_EPOCH=$(($PRUNE_START_EPOCH + 13  ))
RUNNAME=${NETNAME}_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_lambda${L1_LAMBDA}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=08_April_2018_Sunday_14_50_19resnet29_slimming_50_lambda0.00001_fixedmaxpool_endofcycle_checkpoint_0  --disable_l1_reg_after_epoch  --l1_reg_final_epoch=$REG_LAST_EPOCH --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=1 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=l1reg_context_slimming  --enable_l1reg  --l1reg_lambda=$L1_LAMBDA --squeezenet_bypass_first_last --prune_layer_mode=global --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle  --enable_pruning --prune_target_frac=0.$PRUNE_TARG_DECIMAL --autocalc_prune_unit  --prune_warmup_epochs=$PRUNE_START_EPOCH  --fire_skip_mode=zero_pad --log_to_file --log_file_name=log_for_$RUNNAME  --cuda 

PRUNE_TARG_DECIMAL=80
RUNTYPE=slimming
NETNAME=resnet29
PRUNE_START_EPOCH=0
L1_LAMBDA=0.00001
REG_LAST_EPOCH=$(($PRUNE_START_EPOCH + 13  ))
RUNNAME=${NETNAME}_${RUNTYPE}_${PRUNE_TARG_DECIMAL}_lambda${L1_LAMBDA}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=cifar10 --resume_mode=standard --res_file=08_April_2018_Sunday_14_50_19resnet29_slimming_50_lambda0.00001_fixedmaxpool_endofcycle_checkpoint_0  --disable_l1_reg_after_epoch  --l1_reg_final_epoch=$REG_LAST_EPOCH --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=1 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=l1reg_context_slimming  --enable_l1reg  --l1reg_lambda=$L1_LAMBDA --squeezenet_bypass_first_last --prune_layer_mode=global --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle  --enable_pruning --prune_target_frac=0.$PRUNE_TARG_DECIMAL --autocalc_prune_unit  --prune_warmup_epochs=$PRUNE_START_EPOCH  --fire_skip_mode=zero_pad --log_to_file --log_file_name=log_for_$RUNNAME  --cuda 


