#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=20000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:30
#SBATCH --output=%N-%jfactortest.out
module purge
module load python/3.6.3
source ~/env1/bin/activate
ARGFILE=./sh/argfiles/ultrasmall
RESUME_FILENAME=21_March_2018_Wednesday_23_40_27anneal-medium_endofcycle_checkpoint_3
SAVE_PREFIX=factor_test
REPORT_FILENAME=./reports/small_re_factorize

python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --factorize_trained  --proxy_context_type=stdfactorize_context --cuda 

#python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}_prune_$PRUNE_PCT   --use_custom_test_data_file --custom_test_data_file="./local_data/cifar/train_data" --test_report_filename=$REPORT_FILENAME  --cuda

#python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar/train_data



