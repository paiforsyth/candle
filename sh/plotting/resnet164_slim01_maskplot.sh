ARGFILE=./sh/argfiles/resnet_164_slimming
RESUME_FILENAME=05_April_2018_Thursday_16_17_25resnet164_slim03_most_recent


python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX  --resume_mode=standard  --res_file=${RESUME_FILENAME}    --plot_unpruned_masks --plot_title="Resnet 164 after 90% reduction in flops via network slimming. "




