python -m examples.lab   --save_prefix=mnist_mlp_l0 --dataset_for_classification=mnist --model_type=squeezenet   --squeezenet_out_dim=10 --squeezenet_in_channels=1  --squeezenet_mode=mnist_mlp   --batch_size=128 --num_epochs=300 --optimizer=adam   --init_lr=0.001  --proxy_context_type=filter_prune_context   --save_every_epoch --use_no_grad --report_unpruned     --grad_norm_clip=50  --show_network_strucutre_every_epoch  --hz_lasso_enable  --hz_lasso_at_epoch=0   --hz_lasso_num_samples=10  --hz_lasso_target_prop=0.5  --hz_lasso_use_train_loader  --squeezenet_use_mnist_mlp  # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   


#actually this doesnt work.  Need to use convolutional network to use hzpruning
