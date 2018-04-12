ARGFILE=./sh/argfiles/resnet_164_slimming
RESUME_FILENAME=07_April_2018_Saturday_13_33_42resnet164_slimming_90_fixedmaxpool_endofcycle_checkpoint_0  
SAVE_PREFIX=res_net_164_prune_slimming
PRUNE_PCT=5
REPORT_FILENAME=./reports/medium_res_net_164_naive_prune_slimming_$PRUNE_PCT

python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX  --resume_mode=standard  --res_file=$RESUME_FILENAME  --proxy_context_type=l1reg_context_slimming  --enable_l1reg  --squeezenet_bypass_first_last --prune_layer_mode=global --prune_trained --prune_trained_pct=$PRUNE_PCT --cuda

python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}_prune_$PRUNE_PCT   --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"


python -m examples.lab --save_prefix=$SAVE_PREFIX $(cat $ARGFILE) --resume_mode=standard  --res_file=${RESUME_FILENAME}_prune_$PRUNE_PCT   --count_multiplies    

