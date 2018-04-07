ARGFILE=./sh/argfiles/resnet_164_slimming
RESUME_FILENAME=06_April_2018_Friday_19_34_03resnet164_slimming_10_fixedmaxpool_endofcycle_checkpoint_0



python -m examples.lab $(cat $ARGFILE) --save_prefix=$SAVE_PREFIX  --resume_mode=standard  --res_file=${RESUME_FILENAME}  --count_multiplies --output_level=debug # --print_model # --plot_unpruned_masks --plot_title="Resnet 164 after 90% reduction in flops via network slimming. "




