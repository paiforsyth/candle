python -m examples.lab   --save_prefix=lenet --dataset_for_classification=mnist --model_type=squeezenet   --squeezenet_out_dim=10 --squeezenet_in_channels=1  --squeezenet_mode=mnist_lenet_simp   --batch_size=128 --num_epochs=300 --optimizer=adam   --init_lr=0.001  --proxy_context_type=filter_prune_context  --save_every_epoch --use_no_grad    --show_nonzero_masks_every_epoch  --grad_norm_clip=50     --hz_lasso_enable  --hz_lasso_at_epoch=0   --hz_lasso_num_samples=10  --hz_lasso_target_prop=0.5  --hz_lasso_use_train_loader --hz_lasso_solve_for_weights  --show_nonzero_masks_every_epoch   # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   



