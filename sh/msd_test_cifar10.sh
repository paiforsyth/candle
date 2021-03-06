#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=20000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:00
#SBATCH --output=%N-%jmsd.out
python -m examples.lab --save_prefix=msd --dataset_for_classification=cifar10 --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3 --squeezenet_skip_conv1 --squeezenet_mode=msd_fire --squeezenet_msd_growth_rate=10 --squeezenet_msd_num_scales=3 --squeezenet_disable_pooling   --squeezenet_num_fires=14 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=0    --batch_size=64 --num_epochs=120 --optimizer=sgd --sgd_momentum=0.9  --init_lr=0.2 --lr_scheduler=epoch_anneal --sgd_weight_decay=0.0005  --epoch_anneal_numcycles=1   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=no_context     --save_every_epoch --use_no_grad  --count_multiplies_every_cycle --cuda   




