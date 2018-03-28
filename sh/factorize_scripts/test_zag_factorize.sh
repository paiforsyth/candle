
ARGFILE=./sh/argfiles/zag_small
RESUME_FILENAME=21_March_2018_Wednesday_23_40_27anneal-medium_endofcycle_checkpoint_3
SAVE_PREFIX=res_net_smallm
PRUNE_PCT=10
REPORT_FILENAME=./reports/small_re_factorize

python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --factorize_trained  --proxy_context_type=stdfactorize_context --factorize_svd_rank_prop=0.1 #--cuda 

#python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}_prune_$PRUNE_PCT   --use_custom_test_data_file --custom_test_data_file="./local_data/cifar/train_data" --test_report_filename=$REPORT_FILENAME  --cuda

#python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar/train_data



