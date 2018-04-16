#!/bin/bash
RUNTYPE=vgg_cifar100
RUNNAME=${RUNTYPE}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=minicifar10 --resume_mode=standard --res_file=16_April_2018_Monday_17_05_48vgg_cifar100_fixedmaxpool_most_recent --adjust_out_dim_after_loading --new_out_dim=10  --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=100 --squeezenet_in_channels=3  --squeezenet_mode=vgg_fire --squeezenet_final_mode=linear --squeezenet_final_side_length=8 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=2   --squeezenet_num_fires=6 --squeezenet_pool_interval=2 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=1 --new_final_linear  --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_weight_decay=0.0001    --sgd_momentum=0.9 --init_lr=0.01  --epoch_anneal_numcycles=1  --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context     --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle  --cuda # --log_to_file --log_file_name=log_for_$RUNNAME --cuda #--cuda # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   



