RUNTYPE=big_louizos
NETTYPE=wrn2810
ARGFILE=./sh/argfiles/louizos
RESUME_FILENAME=09_April_2018_Monday_01_13_56_endofcycle_checkpoint_1


SAVE_PREFIX=${NETTYPE}_${RUNTYPE}_test_fixedmaxpool
REPORT_FILENAME=./reports/${NETTYPE}_${RUNTYPE}_${PTARG}__test_fixedmaxpool


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX --mode=test --resume_mode=standard  --res_file=${RESUME_FILENAME}    --test_report_filename=$REPORT_FILENAME  --cuda

python -m examples.lab --validate_fr  --validate_fr_reportfile=$REPORT_FILENAME  --validate_fr_truthfile=local_data/cifar10/cifar-10-batches-py/test_batch --validate_fr_truthfiletype="pickle_dict"



