#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=20000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:00
#SBATCH --output=%N-%jmult_test.out
module purge
module load python/3.6.3
source ~/env1/bin/activate
python -m examples.lab --save_prefix=mult_test --dataset_for_classification=cifar_challenge --model_type=squeezenet --squeezenet_out_dim=100 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=4 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=16 --squeezenet_freq=10 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=1 --squeezenet_pool_interval=10 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=4 --squeezenet_pooling_count_offset=0 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply   --batch_size=64 --num_epochs=300 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=5   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=l0reg_context  --enable_l0reg   --save_every_epoch    --count_multiplies  --use_all_params  --output_level=debug

#expected result= 
#3*3*3*32*32*4 +4*1*1*32*32*4 +4/4*3*3*32*32*4 +4*1*1*32*32*16 +16*1*1*32*32*100
#110592        +16384         +36864           +65536          +1638400
#=1867776

