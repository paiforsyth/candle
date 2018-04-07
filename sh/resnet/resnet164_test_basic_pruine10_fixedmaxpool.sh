ARGFILE=./sh/argfiles/resnet_164_no_first_last
RESUME_FILENAME=
SAVE_PREFIX=res_net_164_basic_prune_10_test_fixedmaxpool
REPORT_FILENAME=./reports/res_net_basic_prune_10_test_fixedmaxpool


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}    --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"



