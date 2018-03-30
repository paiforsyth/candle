
ARGFILE=./sh/argfiles/zag_small
RESUME_FILENAME=29_March_2018_Thursday_11_56_05zag_branchy_checkpoint_0
SAVE_PREFIX=zag_branch_eval
ENTROPY_THRESHOLD=0.5
REPORT_FILENAME=./reports/zag_branch_$ENTROPY_THRESHOLD


python -m examples.lab $(cat $ARGFILE)   --proxy_context_type=no_context  --save_prefix=$SAVE_PREFIX  --resume_mode=standard  --get_forking_props_on_val  --res_file=${RESUME_FILENAME}  --count_multiplies --cuda  --squeezenet_use_forking --squeezenet_fork_after_chunks 0 1  --squeezenet_num_layer_chunks=3 --squeezenet_use_non_default_layer_splits --squeezenet_layer_splits 0 3 12 --squeezenet_fork_early_exit   --squeezenet_fork_entropy_threshold $ENTROPY_THRESHOLD   --batch_size=1 --cuda 

python -m examples.lab $(cat $ARGFILE)   --proxy_context_type=no_context  --save_prefix=$SAVE_PREFIX  --resume_mode=standard --mode=test --use_val_as_test   --res_file=${RESUME_FILENAME}  --test_report_filename=$REPORT_FILENAME   --squeezenet_use_forking --squeezenet_fork_after_chunks 0 1  --squeezenet_num_layer_chunks=3 --squeezenet_use_non_default_layer_splits --squeezenet_layer_splits 0 3 12 --squeezenet_fork_early_exit   --squeezenet_fork_entropy_threshold $ENTROPY_THRESHOLD --batch_size=1 --cuda 



python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/data_batch_1  --validate_fr_truthfiletype="pickle_dict"





