ARGFILE=./sh/argfiles/medium_res_net
RESUME_FILENAME=21_March_2018_Wednesday_23_40_27anneal-medium_endofcycle_checkpoint_3
SAVE_PREFIX=res_net_anneal_medium
PRUNE_PCT=10
REPORT_FILENAME=./reports/medium_res_net_prune_$PRUNE_PCT


python -m examples.lab --save_prefix=$SAVE_PREFIX $(cat $ARGFILE) --resume_mode=standard  --res_file=${RESUME_FILENAME} --count_multiplies    --output_level=debug
