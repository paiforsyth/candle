
ARGFILE=./sh/argfiles/zag_small
RESUME_FILENAME=29_March_2018_Thursday_11_56_05zag_branchy_checkpoint_0
SAVE_PREFIX=zag_branch_eval
REPORT_FILENAME=./reports/zag_branch



python -m examples.lab $(cat $ARGFILE)   --proxy_context_type=no_context  --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --get_forking_props_on_val  --res_file=${RESUME_FILENAME}  --count_multiplies --cuda  --squeezenet_use_forking --squeezenet_fork_after_chunks 0 1  --squeezenet_num_layer_chunks=3 --squeezenet_use_non_default_layer_splits --squeezenet_layer_splits 0 3 12 --squeezenet_fork_early_exit   --squeezenet_fork_entropy_threshold 0.5    --cuda 




#retrain

