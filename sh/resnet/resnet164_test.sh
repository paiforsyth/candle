ARGFILE=./sh/argfiles/resnet_164
RESUME_FILENAME=04_April_2018_Wednesday_13_00_36resnet164_endofcycle_checkpoint_0
SAVE_PREFIX=res_net_164_test
REPORT_FILENAME=./reports/res_net_164_test


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}    --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"



