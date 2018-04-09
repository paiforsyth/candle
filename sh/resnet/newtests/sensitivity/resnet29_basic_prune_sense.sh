RUNTYPE=sensitivity_group_prune
NETTYPE=resnet29
ARGFILE=./sh/argfiles/${NETTYPE}_group
RESUME_FILENAME=09_April_2018_Monday_02_17_27_endofcycle_checkpoint_0 
SAVE_PREFIX=${NETTYPE}_${RUNTYPE}__test_fixedmaxpool


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX  --resume_mode=standard --prune_unit=10  --res_file=${RESUME_FILENAME}  --sensitivity_report  # --cuda


