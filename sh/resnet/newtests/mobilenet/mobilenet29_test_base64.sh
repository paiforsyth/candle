RUNTYPE=moblenet29
NETTYPE=mobilenet29_base64
ARGFILE=./sh/argfiles/${NETTYPE}
RESUME_FILENAME=12_April_2018_Thursday_03_34_36mobile_resnet29_group_prune_base64_fixedmaxpool_most_recent
SAVE_PREFIX=${NETTYPE}_${RUNTYPE}_test_fixedmaxpool
REPORT_FILENAME=./reports/${NETTYPE}_${RUNTYPE}_${PTARG}__test_fixedmaxpool


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}    --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"



