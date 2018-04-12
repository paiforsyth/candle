for PRUNE_PCT in 0 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70; do
ARGFILE=./sh/argfiles/resnet29_filter_prune
RESUME_FILENAME=11_April_2018_Wednesday_20_55_15resnet29_hz_lasso_70_before_hz_lasso
SAVE_PREFIX=res_net29_hz
REPORT_FILENAME=./reports/res29_hzprune_$PRUNE_PCT

python -W ignore -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX  --resume_mode=standard  --res_file=$RESUME_FILENAME   --prune_trained --prune_trained_pct=$PRUNE_PCT --prune_trained_hz --hz_lasso_solve_for_weights  --cuda  --output_level=warning

python  -W ignore -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}_prune_$PRUNE_PCT   --test_report_filename=$REPORT_FILENAME  --cuda --output_level=warning

python  -W ignore -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"  --output_level=warning


python  -W ignore -m examples.lab --save_prefix=$SAVE_PREFIX $(cat $ARGFILE) --resume_mode=standard  --res_file=${RESUME_FILENAME}_prune_$PRUNE_PCT   --count_multiplies    --output_level=warning

done