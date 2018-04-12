PRUNE_TARG_DECIMAL=90
RUNTYPE=group_prune
NETNAME=mobile_resnet29
PRUNE_START_EPOCH=102
for BASE in 16 32 64 128 256; do 
RUNNAME=${NETNAME}_${RUNTYPE}_base${BASE}_fixedmaxpool
python -m examples.lab --save_prefix=$RUNNAME --dataset_for_classification=cifar10  --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=shuffle_fire --squeezenet_shuffle_fire_g1=1 --squeezenet_shuffle_fire_g2=1 --squeezenet_shuffle_fire_dont_wrap_sepconv --squeezenet_pool_interval_mode=multiply  --squeezenet_base=$BASE --squeezenet_freq=3 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=1 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=100 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context  --squeezenet_bypass_first_last --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle  --enable_pruning --prune_target_frac=0.$PRUNE_TARG_DECIMAL --autocalc_prune_unit  --prune_warmup_epochs=$PRUNE_START_EPOCH  --fire_skip_mode=zero_pad --log_to_file --log_file_name=log_for_$RUNNAME --cuda  #--param_report #--mod_report  #--cuda 

done
