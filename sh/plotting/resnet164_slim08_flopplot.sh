ARGFILE=./sh/argfiles/resnet_164_slimming
RESUME_FILENAME=04_April_2018_Wednesday_20_23_17resnet164_slim08_most_recent


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX  --resume_mode=standard  --res_file=${RESUME_FILENAME}    --plot_flop_reduction_by_layer --plot_title="Resnet 164 after 20% reduction in flops via network slimming. "




