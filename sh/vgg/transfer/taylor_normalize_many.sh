#!/bin/bash

for PRUNE_TARG_DECIMAL in 99 95 92 90 80 70 60 50 40 30 10 20 10; do
echo $PRUNE_TARG_DECIMAL
for COUNT in 1 2 3 4 5; do


RUNTYPE=vgg_cifar100

LAYER_MODE=global
PRUNE_STRAT=taylor
PRUNE_ABS_UNIT=1
RUNNAME=vgg_${RUNTYPE}_${PRUNE_STRAT}_${LAYER_MODE}_${PRUNE_TARG_DECIMAL}_fixedmaxpool_count_${COUNT}
python  -W ignore -m examples.lab --save_prefix=$RUNNAME  --group_prune_strategy=$PRUNE_STRAT  --global_prune_normalization=by_layer  --maintain_abs_deriv_sum --prune_layer_mode=$LAYER_MODE    --group_prune_strategy=$PRUNE_STRAT  --global_prune_normalization=by_layer  --dataset_for_classification=minicifar10 --resume_mode=standard --res_file=16_April_2018_Monday_17_05_48vgg_cifar100_fixedmaxpool_most_recent --adjust_out_dim_after_loading --new_out_dim=10  --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=100 --squeezenet_in_channels=3  --squeezenet_mode=vgg_fire --squeezenet_final_mode=linear --squeezenet_final_side_length=8 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=2   --squeezenet_num_fires=6 --squeezenet_pool_interval=2 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=1 --new_final_linear  --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=20000 --optimizer=sgd --sgd_weight_decay=0.0001    --sgd_momentum=0.9 --init_lr=0.02  --epoch_anneal_numcycles=1  --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context      --use_no_grad --report_unpruned   --use_no_grad --report_unpruned   --enable_pruning  --prune_target_frac=0.$PRUNE_TARG_DECIMAL --prune_warmup_epochs=200  --prune_unit=$PRUNE_ABS_UNIT --prune_absolute   --count_multiplies_every_cycle --terminate_after_pruning  --report_test_error_at_end  --iterations_after_pruning=200   --count_multiplies_every_cycle  --cuda  --log_to_file --log_file_name=log_for_$RUNNAME --short_run_report --cuda #--cuda # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   

done
done
