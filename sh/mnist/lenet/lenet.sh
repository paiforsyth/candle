python -m examples.lab   --save_prefix=lenet --dataset_for_classification=mnist --model_type=squeezenet   --squeezenet_out_dim=10 --squeezenet_in_channels=1  --squeezenet_mode=mnist_lenet   --batch_size=128 --num_epochs=300 --optimizer=adam   --init_lr=0.001  --proxy_context_type=no_context   --save_every_epoch --use_no_grad    --show_nonzero_masks_every_epoch  --grad_norm_clip=50       # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   



