RUNTYPE=basic_prune
PTARG=70
ARGFILE=./sh/argfiles/resnet_164_${RUNTYPE}
RESUME_FILENAME=08_April_2018_Sunday_09_09_56_most_recent
SAVE_PREFIX=res_net_164_${RUNTYPE}_${PTARG}_test_fixedmaxpool
REPORT_FILENAME=./reports/res_net_${RUNTYPE}_${PTARG}_test_fixedmaxpool


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}    --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"



