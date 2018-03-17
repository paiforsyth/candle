#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=20000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-24:00
#SBATCH --output=%N-%jbnn_basic.out
#module purge
#module load python/3.6.3
#source ~/env1/bin/activate
python -m examples.lab --save_prefix=bnn_basic --dataset_for_classification=cifar10 --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=bnnfire --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=2 --squeezenet_num_fires=6 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=64 --batch_size=64 --num_epochs=300 --optimizer=sgd --sgd_momentum=0.9 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=5   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase  --save_every_epoch --squeezenet_final_act_mode=disable  --proxy_context_type=tanhbinarize_context --squeezenet_disable_pooling --squeezenet_bnn_pooling --squeezenet_pool_interval=2 --classification_loss_type=square_hinge  --cuda   



