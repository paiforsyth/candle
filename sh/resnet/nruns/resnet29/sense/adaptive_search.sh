#!/usr/bin/env bash
#FRACS='01 10 20 30 40 50 60 70 80 90 90'
FRACS=01
RUNTYPE=sensitivity_adaptive_group_prune
NETTYPE=resnet29
ARGFILE=./sh/argfiles/${NETTYPE}_group
RESUME_FILENAME=09_April_2018_Monday_02_17_27_endofcycle_checkpoint_0 
LR_FRAC=05

for P_FRAC in $FRACS
do
    RUNNAME=${NETTYPE}_${RUNTYPE}_p${P_FRAC}_lr${LR_FRAC}_test_fixedmaxpool
    SAVE_PREFIX=$RUNNAME
    python -m examples.lab --dataset_for_classification=cifar10 --model_type=squeezenet --squeezenet_bypass_first_last --squeezenet_out_dim=10 --squeezenet_in_channels=3 --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply --squeezenet_base=64 --squeezenet_freq=3 --squeezenet_freq_offset=0 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=1 --squeezenet_pool_interval_mode=multiply --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.$LR_FRAC --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1 --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context --save_every_epoch --use_no_grad --report_unpruned --count_multiplies_every_cycle --fire_skip_mode=zero_pad  --save_prefix=$SAVE_PREFIX  --resume_mode=standard --enable_pruning   --res_file=${RESUME_FILENAME}  --prune_target_frac=0.$P_FRAC  --enable_pruning --prune_warmup_epoch=0  --autocalc_prune_unit --prune_phase_duration=40 --sense_adaptive_pruning  --log_to_file --log_file_name=log_for_$RUNNAME --cuda
done
