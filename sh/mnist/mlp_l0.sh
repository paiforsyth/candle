python -m examples.lab   --save_prefix=mnist_mlp_l0 --dataset_for_classification=mnist --model_type=squeezenet   --squeezenet_out_dim=10 --squeezenet_in_channels=1  --squeezenet_mode=mnist_mlp   --batch_size=128 --num_epochs=300 --optimizer=adam   --init_lr=0.001  --proxy_context_type=l0reg_context --enable_l0reg --l0reg_lambda=0.0000001 --use_all_params  --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle --cifar_random_erase  --grad_norm_clip=50  --show_network_strucutre_every_epoch --cuda  # --log_to_file --log_file_name=log_for_mnist_mlp  --squeezenet_use_mnist_mlp  # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   



