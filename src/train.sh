python3 train.py --exp_num 4000 --model_name SocialVRNN --n_mixtures 1 --output_pred_state_dim 4 --scenario real_world/amsterdam_canals --gpu false --prev_horizon 7 --prediction_horizon 20 --dt 1.0 --truncated_backprop_length 20 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --n_other_agents 6