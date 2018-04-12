ARGFILE=./sh/argfiles/resnet_164_basic_prune
RESUME_FILENAME=07_April_2018_Saturday_13_31_11_endofcycle_checkpoint_0   #04_April_2018_Wednesday_13_00_36resnet164_endofcycle_checkpoint_0
SAVE_PREFIX=res_net_164_prune
PRUNE_PCT=5
REPORT_FILENAME=./reports/medium_res_net_164_naive_prune_$PRUNE_PCT

python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX  --resume_mode=standard  --res_file=$RESUME_FILENAME   --prune_trained --prune_trained_pct=$PRUNE_PCT --cuda  --output_level=warning

python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}_prune_$PRUNE_PCT   --test_report_filename=$REPORT_FILENAME  --cuda --output_level=warning

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"  --output_level=warning


python -m examples.lab --save_prefix=$SAVE_PREFIX $(cat $ARGFILE) --resume_mode=standard  --res_file=${RESUME_FILENAME}_prune_$PRUNE_PCT   --count_multiplies    --output_level=warning

