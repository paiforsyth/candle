#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=20000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:00
#SBATCH --output=%N-%jstdmedium.out
module purge
module load python/3.6.3
source ~/env1/bin/activate
python -m examples.lab --save_prefix=anneal-medium --dataset_for_classification=cifar_challenge --model_type=squeezenet --squeezenet_out_dim=100 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=4 --squeezenet_freq=10 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=2 --squeezenet_pool_interval=1 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=4 --squeezenet_pooling_count_offset=1 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=64 --num_epochs=300 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=5   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context  --save_every_epoch    --count_multiplies    --output_level=debug --squeezenet_allow_pooling_after_first_fire
#dimesnions when there is only one fire
#=3*3*3*32*32*4+  4*1*1*32*32*1 +1*3*3*32*32*1 +1*1*1*32*32*4 +4*1*1*32*32*100
#110592+ 4096 +9216 + 4096 +409600
#=110592 +17408 + 409600
#=537600
#If we insert another residual unit and seperate the two residual units with a max pool, then since the max pool halves the dimensions of the image, the second residual unit contributes (1/4)*17408.  
#Also, since the final convolution is now applied to a shrunk image, it contributes 1/4*409600.  #Thus the total is
#=110592 +(5/4)*17408 + (1/4)*409600
#=234752

