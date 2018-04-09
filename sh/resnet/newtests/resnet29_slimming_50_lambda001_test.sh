RUNTYPE=slimming
PTARG=50
NETTYPE=resnet29_slimming
LAMBDA=001
ARGFILE=./sh/argfiles/${NETTYPE}_${RUNTYPE}
RESUME_FILENAME=08_April_2018_Sunday_17_55_41resnet29_slimming_50_lambda0.001_fixedmaxpool_most_recent
SAVE_PREFIX=${NETTYPE}_${RUNTYPE}_${PTARG}_lambda${LAMBDA}_test_fixedmaxpool
REPORT_FILENAME=./reports/${NETTYPE}_${RUNTYPE}_${PTARG}_lambda${LAMBDA}_test_fixedmaxpool


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}    --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"



