#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=20000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:00
#SBATCH --output=%N-%jzag.out
#WRN 28-> 27 conv units plus the initial conv unit.  -> 9 per section
python -m examples.lab --save_prefix=zag --dataset_for_classification=cifar10 --model_type=squeezenet --squeezenet_bypass_first_last --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=zag_fire  --squeezenet_pool_interval_mode=multiply  --squeezenet_base=160 --squeezenet_freq=4   --squeezenet_num_fires=12 --squeezenet_pool_interval=4 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=1 --squeezenet_max_pool_size=2   --batch_size=128 --num_epochs=200 --optimizer=sgd --sgd_momentum=0.9  --init_lr=0.2 --lr_scheduler=epoch_anneal --sgd_weight_decay=0.0005  --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=no_context     --save_every_epoch --use_no_grad --cuda # count_multiplies  # --param_report  #--mod_report  #--count_multiplies # --count_multiplies_every_cycle --cuda   



