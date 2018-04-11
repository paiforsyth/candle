RUNTYPE=filter_prune
PTARG=10
NETTYPE=resnet29
ARGFILE=./sh/argfiles/${NETTYPE}_filter_prune
RESUME_FILENAME=10_April_2018_Tuesday_15_17_16resnet29_filter_prune_10_most_recent


SAVE_PREFIX=${NETTYPE}_${RUNTYPE}_${PTARG}__test_fixedmaxpool
REPORT_FILENAME=./reports/${NETTYPE}_${RUNTYPE}_${PTARG}__test_fixedmaxpool


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}    --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"



