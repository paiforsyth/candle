#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=20000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=3-00:00
#SBATCH --output=%N-%jsmall_shufflenet.out
module purge
module load python/3.6.3
source ~/env1/bin/activate
python -m examples.lab --save_prefix=small_shufflenet_cifar10 --dataset_for_classification=cifar10 --model_type=squeezenet --squeezenet_sr=0.25 --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=shuffle_fire --squeezenet_pool_interval_mode=multiply  --squeezenet_base=224 --squeezenet_freq=9 --squeezenet_num_fires=24 --squeezenet_pool_interval=9 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=224 --squeezenet_pooling_count_offset=0 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply  --batch_size=64 --num_epochs=120 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=no_context  --save_every_epoch --use_no_grad --count_multiplies_every_cycle --cuda --count_multiplies    --output_level=debug    



