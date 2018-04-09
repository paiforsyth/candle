RUNTYPE=moblenet29
NETTYPE=mobilenet29
ARGFILE=./sh/argfiles/${NETTYPE}
RESUME_FILENAME=09_April_2018_Monday_12_10_52mobile_resnet29_group_prune_90_fixedmaxpool_endofcycle_checkpoint_0
SAVE_PREFIX=${NETTYPE}_${RUNTYPE}_test_fixedmaxpool
REPORT_FILENAME=./reports/${NETTYPE}_${RUNTYPE}_${PTARG}__test_fixedmaxpool


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}    --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"



