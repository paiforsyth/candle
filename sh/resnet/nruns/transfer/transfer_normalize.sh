#!/bin/bash
for PRUNE_TARG_DECIMAL in 90 80 70 60 50 40 30 10 20 10 5 1; do
PRUNE_ABS_UNIT=10
PRUNE_STRAT=taylor
LAYER_MODE=global
RUNTYPE=basic_prune_transfer
RUNNAME=resnet164_${RUNTYPE}_${PRUNE_STRAT}_${LAYER_MODE}_${PRUNE_TARG_DECIMAL}_fixedmaxpool
python -m examples.lab  --group_prune_strategy=$PRUNE_STRAT --maintain_abs_deriv_sum --prune_layer_mode=$LAYER_MODE   --save_prefix=$RUNNAME --dataset_for_classification=minicifar10  --resume_mode=standard --res_file=15_April_2018_Sunday_03_18_39_most_recent --adjust_out_dim_after_loading  --new_out_dim=10  --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=100 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=18 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=54 --squeezenet_pool_interval=18 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=0   --squeezenet_pool_interval_mode=multiply   --batch_size=100 --num_epochs=10000 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001  --init_lr=0.05 --lr_scheduler=none --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context     --save_every_epoch --use_no_grad --report_unpruned   --enable_pruning  --prune_target_frac=0.$PRUNE_TARG_DECIMAL --prune_warmup_epochs=50  --prune_unit=$PRUNE_ABS_UNIT --prune_absolute   --count_multiplies_every_cycle --terminate_after_pruning   --fire_skip_mode=zero_pad --eval_interval=10  --report_test_error_at_end --cuda --log_to_file --log_file_name=log_for_$RUNNAME --cuda #--cuda # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   


done
