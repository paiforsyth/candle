ARGFILE=./sh/argfiles/medium_resnext
RESUME_FILENAME=17_March_2018_Saturday_07_14_11anneal-medium_endofcycle_checkpoint_1 
SAVE_PREFIX=anneal_medium
PRUNE_PCT=10
REPORT_FILENAME=./reports/medium_resnext_prune_$PRUNE_PCT

python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX  --resume_mode=standard  --res_file=$RESUME_FILENAME   --prune_trained --prune_trained_pct=$PRUNE_PCT --cuda


