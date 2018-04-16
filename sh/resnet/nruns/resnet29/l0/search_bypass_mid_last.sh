#!/bin/bash
RUNTYPE=l0_reg
NETTYPE=resnet29
for LAM in 0.0001 0.00001 0.000001 0.0000001; do
RUNNAME=${NETTYPE}_${RUNTYPE}_bypas_second_third_lambda_${LAM}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=minicifar10 --model_type=squeezenet    --squeezenet_next_fire_bypass_second  --squeezenet_bypass_first_last  --squeezenet_next_fire_bypass_third  --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1  --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=3   --squeezenet_num_fires=9 --squeezenet_sr=0.25 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=1   --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --squeezenet_num_layer_chunks=1  --proxy_context_type=l0reg_context --enable_l0reg --l0reg_lambda=$LAM  --optimizer=sgd --sgd_momentum=0.9   --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last    --use_all_params  --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle --cifar_random_erase --fire_skip_mode=zero_pad --grad_norm_clip=50   --report_test_error_at_end  --show_nonzero_masks_every_epoch  --cuda   --log_to_file --log_file_name=log_for_$RUNNAME  # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   
done

