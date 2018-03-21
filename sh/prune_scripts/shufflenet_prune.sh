ARGFILE=./sh/argfiles/shufflenet
RESUME_FILENAME=20_March_2018_Tuesday_17_36_33shufflenet_endofcycle_checkpoint_1 
SAVE_PREFIX=shufflenet
PRUNE_PCT=10
REPORT_FILENAME=./reports/shufflenet_prune_$PRUNE_PCT

python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX  --resume_mode=standard  --res_file=$RESUME_FILENAME   --prune_trained --prune_trained_pct=$PRUNE_PCT --cuda

python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}_prune_$PRUNE_PCT   --use_custom_test_data_file --custom_test_data_file="./local_data/cifar/train_data" --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar/train_data

