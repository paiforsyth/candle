for PRUNE_PCT  in 5 15 20 40 60; do   #in 10 30 50 70; do 
ARGFILE=./sh/argfiles/resnet29_group
RESUME_FILENAME=11_April_2018_Wednesday_23_26_51_most_recent
SAVE_PREFIX=res_net_29_group_prune_recalc
REPORT_FILENAME=./reports/medium_res_net_164_naive_prune_$PRUNE_PCT

python -W ignore -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX  --resume_mode=standard  --res_file=$RESUME_FILENAME   --prune_trained --prune_trained_pct=$PRUNE_PCT --cuda --recalc_weights_after_prune_trained --output_level=warning --short_test_report

python  -W ignore -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}_prune_$PRUNE_PCT   --test_report_filename=$REPORT_FILENAME  --cuda --output_level=warning  --short_test_report

python  -W ignore -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"  --output_level=warning  --short_test_report


python  -W ignore -m examples.lab --save_prefix=$SAVE_PREFIX $(cat $ARGFILE) --resume_mode=standard  --res_file=${RESUME_FILENAME}_prune_$PRUNE_PCT   --count_multiplies    --output_level=warning   --short_test_report

done
