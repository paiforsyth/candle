RUNTYPE=random
PTARG=80
NETTYPE=resnet29
LAMBDA=00001
ARGFILE=./sh/argfiles/${NETTYPE}_group
RESUME_FILENAME=09_April_2018_Monday_09_00_13resnet29_random_random_80_fixedmaxpool_most_recent
SAVE_PREFIX=${NETTYPE}_${RUNTYPE}_${PTARG}_lambda${LAMBDA}_test_fixedmaxpool
REPORT_FILENAME=./reports/${NETTYPE}_${RUNTYPE}_${PTARG}_lambda${LAMBDA}_test_fixedmaxpool


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}    --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"



