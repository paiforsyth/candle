RUNTYPE=group_prune_from_scratch
PTARG=01
NETTYPE=resnet29
LAMBDA=00001
ARGFILE=./sh/argfiles/${NETTYPE}_group
RESUME_FILENAME=saved_models/09_April_2018_Monday_02_17_27_endofcycle_checkpoint_0
SAVE_PREFIX=${NETTYPE}_${RUNTYPE}_${PTARG}_lambda${LAMBDA}_test_fixedmaxpool
REPORT_FILENAME=./reports/${NETTYPE}_${RUNTYPE}_${PTARG}_lambda${LAMBDA}_test_fixedmaxpool_noprune


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}    --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"



