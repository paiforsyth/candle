#!/bin/bash
RUNTYPE=basic_cifar100
RUNNAME=resnet110_${RUNTYPE}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=cifar_challenge --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_zag_fire_dropout=0  --squeezenet_out_dim=100 --squeezenet_in_channels=3  --squeezenet_mode=zag_fire  --squeezenet_zag_dont_bypass_last  --squeezenet_pool_interval_mode=multiply  --squeezenet_base=16 --squeezenet_freq=18 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=54 --squeezenet_pool_interval=18 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=0   --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001  --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context     --save_every_epoch --use_no_grad --report_unpruned   --count_multiplies_every_cycle --fire_skip_mode=zero_pad  --log_to_file --log_file_name=log_for_$RUNNAME --cuda #--cuda # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   



