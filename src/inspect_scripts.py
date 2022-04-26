import numpy as np
import os
import pickle
#from src.data_utils.Trajectory import *
import sys
import matplotlib.pyplot as plt

sys.path.append('../')

#from src.data_utils.AgentContainer import AgentContainer
#import src.data_utils.Support as sup
#from .src.data_utils.Trajectory import *
#import src.data_utils.AgentContainer as ped_cont

if sys.version_info[0] < 3:
    print("Using Python " + str(sys.version_info[0]))
    sys.path.append('../src/data_utils')
    sys.path.append('../src/models')
    import DataHandler as dh
    import DataHandlerLSTM as dhlstm
    from plot_utils import *
    import Support as sup
    from utils import *
else:
    print("Using Python " + str(sys.version_info[0]))
    #from src.data_utils import DataHandler as dh
    from src.data_utils import DataHandlerLSTM as dhlstm
    from src.data_utils.plot_utils import *
    from src.data_utils import Support as sup
    from src.data_utils.utils import *
    import src.data_utils.AgentContainer as ped_cont


home = os.path.expanduser("~")
data_path = "/social_vrnn_fork/data/simulation/"

norm_const_x = 1.0
norm_const_y = 1.0
norm_const_vx = 1.0
norm_const_vy = 1.0
norm_const_heading = 1.0

def parse_args():
	parser = argparse.ArgumentParser(description='LSTM model training')

	parser.add_argument('--model_name',
	                    help='Path to directory that comprises the model (default="model_name").',
	                    type=str, default= "RNN")
	parser.add_argument('--model_path',
	                    help='Path to directory to save the model (default=""../trained_models/"+model_name").',
	                    type=str, default='../trained_models/')
	parser.add_argument('--pretrained_convnet_path',
	                    help='Path to directory that comprises the pre-trained convnet model (default=" ").',
	                    type=str, default=pretrained_convnet_path)
	parser.add_argument('--log_dir',
	                    help='Path to the log directory of the model (default=""../trained_models/"+model_name").',
	                    type=str, default="\log")
	parser.add_argument('--scenario', help='Scenario of the dataset (default="").',
	                    type=str, default=scenario)
	parser.add_argument('--real_world_data', help='Real world dataset (default=True).', type=sup.str2bool,
	                    default=real_world_data)
	parser.add_argument('--data_path', help='Path to directory that saves pickle data (default=" ").', type=str,
	                    default=data_path)
	parser.add_argument('--dataset', help='Dataset pkl file', type=str,
	                    default= scenario + '.pkl')
	parser.add_argument('--data_handler', help='Datahandler class needed to load the data', type=str,
	                    default='LSTM')
	parser.add_argument('--warmstart_model', help='Restore from pretained model (default=False).', type=bool,
	                    default=warmstart_model)
	parser.add_argument('--warm_start_convnet', help='Restore from pretained convnet model (default=False).', type=bool,
	                    default=warm_start_convnet)
	parser.add_argument('--dt', help='Data samplig time (default=0.3).', type=float,
	                    default=dt)
	parser.add_argument('--n_epochs', help='Number of epochs (default=10000).', type=int, default=n_epochs)
	parser.add_argument('--total_training_steps', help='Number of training steps (default=20000).', type=int, default=total_training_steps)
	parser.add_argument('--batch_size', help='Batch size for training (default=32).', type=int, default=batch_size)
	parser.add_argument('--regularization_weight', help='Weight scaling of regularizer (default=0.01).', type=float,
	                    default=regularization_weight)
	parser.add_argument('--keep_prob', help='Dropout (default=0.8).', type=float,
	                    default=keep_prob)
	parser.add_argument('--learning_rate_init', help='Initial learning rate (default=0.005).', type=float,
	                    default=learning_rate_init)
	parser.add_argument('--beta_rate_init', help='Initial beta rate (default=0.005).', type=float,
	                    default=beta_rate_init)
	parser.add_argument('--dropout', help='Enable Dropout', type=sup.str2bool,
	                    default=dropout)
	parser.add_argument('--grads_clip', help='Gridient clipping (default=10.0).', type=float,
	                    default=grads_clip)
	parser.add_argument('--truncated_backprop_length', help='Backpropagation length during training (default=5).',
	                    type=int, default=truncated_backprop_length)
	parser.add_argument('--prediction_horizon', help='Length of predicted sequences (default=10).', type=int,
	                    default=prediction_horizon)
	parser.add_argument('--prev_horizon', help='Previous seq length.', type=int,
	                    default=prev_horizon)
	parser.add_argument('--rnn_state_size', help='Number of RNN / LSTM units (default=16).', type=int,
	                    default=rnn_state_size)
	parser.add_argument('--rnn_state_size_lstm_ped',
	                    help='Number of RNN / LSTM units of the grid lstm layer (default=32).',
	                    type=int, default=rnn_state_size_lstm_ped)
	parser.add_argument('--rnn_state_ped_size',
	                    help='Number of RNN / LSTM units of the grid lstm layer (default=32).',
	                    type=int, default=rnn_state_ped_size)
	parser.add_argument('--rnn_state_size_lstm_grid',
	                    help='Number of RNN / LSTM units of the grid lstm layer (default=32).',
	                    type=int, default=rnn_state_size_lstm_grid)
	parser.add_argument('--rnn_state_size_lstm_concat',
	                    help='Number of RNN / LSTM units of the concatenation lstm layer (default=32).',
	                    type=int, default=rnn_state_size_lstm_concat)
	parser.add_argument('--prior_size', help='prior_size',
	                    type=int, default=prior_size)
	parser.add_argument('--latent_space_size', help='latent_space_size',
	                    type=int, default=latent_space_size)
	parser.add_argument('--x_dim', help='x_dim',
	                    type=int, default=x_dim)
	parser.add_argument('--fc_hidden_unit_size',
	                    help='Number of fully connected layer units after LSTM layer (default=64).',
	                    type=int, default=fc_hidden_unit_size)
	parser.add_argument('--input_state_dim', help='Input state dimension (default=).', type=int,
	                    default=input_state_dim)
	parser.add_argument('--input_dim', help='Input state dimension (default=).', type=float,
	                    default=input_dim)
	parser.add_argument('--output_dim', help='Output state dimension (default=).', type=float,
	                    default=output_dim)
	parser.add_argument('--goal_size', help='Goal dimension (default=).', type=int,
	                    default=2)
	parser.add_argument('--output_pred_state_dim', help='Output prediction state dimension (default=).', type=int,
	                    default=output_pred_state_dim)
	parser.add_argument('--cmd_vector_dim', help='Command control dimension.', type=int,
	                    default=cmd_vector_dim)
	parser.add_argument('--n_mixtures', help='Number of modes (default=).', type=int,
	                    default=n_mixtures)
	parser.add_argument('--pedestrian_vector_dim', help='Number of angular grid sectors (default=72).', type=int,
	                    default=pedestrian_vector_dim)
	parser.add_argument('--pedestrian_vector_state_dim', help='Number of angular grid sectors (default=2).', type=int,
	                    default=pedestrian_vector_state_dim)
	parser.add_argument('--max_range_ped_grid', help='Maximum pedestrian distance (default=2).', type=float,
	                    default=max_range_ped_grid)
	parser.add_argument('--pedestrian_radius', help='Pedestrian radius (default=0.3).', type=float,
	                    default=pedestrian_radius)
	parser.add_argument('--n_other_agents', help='Number of other agents incorporated in the network.', type=int,
	                    default=n_other_agents)
	parser.add_argument('--debug_plotting', help='Plotting for debugging (default=False).', type=sup.str2bool, default=debug_plotting)
	parser.add_argument('--print_freq', help='Print frequency of training info (default=100).', type=int,
	                    default=print_freq)
	parser.add_argument('--save_freq', help='Save frequency of the temporary model during training. (default=20k).',
	                    type=int, default=save_freq)
	parser.add_argument('--exp_num', help='Experiment number', type=int, default=exp_num)
	parser.add_argument('--noise', help='Likelihood? (default=True).', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--agents_on_grid', help='Likelihood? (default=True).', type=sup.str2bool,
	                    default=agents_on_grid)
	parser.add_argument('--normalize_data', help='Normalize? (default=False).', type=sup.str2bool,
	                    default=normalize_data)
	parser.add_argument('--rotated_grid', help='Rotate grid? (default=False).', type=sup.str2bool,
	                    default=rotated_grid)
	parser.add_argument('--centered_grid', help='Center grid? (default=False).', type=sup.str2bool,
	                    default=centered_grid)
	parser.add_argument('--sigma_bias', help='Percentage of the dataset used for trainning', type=float,
	                    default=0)
	parser.add_argument('--submap_width', help='width of occupancy grid', type=int, default=submap_width)
	parser.add_argument('--submap_height', help='height of occupancy grid', type=int, default=submap_height)
	parser.add_argument('--submap_resolution', help='Map resolution.', type=float,
	                    default=submap_resolution)
	parser.add_argument('--min_buffer_size', help='Minimum buffer size (default=1000).', type=int, default=1000)
	parser.add_argument('--max_buffer_size', help='Maximum buffer size (default=100k).', type=int, default=100000)
	parser.add_argument('--max_trajectories', help='maximum number of trajectories to be recorded', type=int, default=30)
	parser.add_argument('--end_to_end', help='End to end trainning.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--predict_positions', help='predict_positions.', type=sup.str2bool,
	                    default=predict_positions)
	parser.add_argument('--gpu', help='Enable GPU training.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--sequence_info', help='Use relative info for other agents.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--others_info', help='Use relative info for other agents.', type=str,
	                    default="none")
	parser.add_argument('--regulate_log_loss', help='Enable GPU training.', type=sup.str2bool,
	                    default=regulate_log_loss)
	parser.add_argument('--diversity_update', help='diversity_update', type=sup.str2bool,
	                    default=diversity_update)
	parser.add_argument('--topics_config', help='yaml file containg subscription topics (default=" ").', type=str,
	                    default='../config/topics.yaml')
	parser.add_argument('--min_pos_x', help='min_pos_x', type=float, default=-1)
	parser.add_argument('--min_pos_y', help='min_pos_y', type=float, default=-1)
	parser.add_argument('--max_pos_x', help='max_pos_x', type=float, default=1)
	parser.add_argument('--max_pos_y', help='max_pos_y', type=float, default=1)
	parser.add_argument('--min_vel_x', help='min_vel_x', type=float, default=-1)
	parser.add_argument('--min_vel_y', help='min_vel_y', type=float, default=-1)
	parser.add_argument('--max_vel_x', help='max_vel_x', type=float, default=1)
	parser.add_argument('--max_vel_y', help='max_vel_y', type=float, default=1)
	parser.add_argument('--sx_vel', help='sx_vel', type=float, default=1)
	parser.add_argument('--sy_vel', help='sy_vel', type=float, default=1)
	parser.add_argument('--sx_pos', help='sx_pos', type=float, default=1)
	parser.add_argument('--sy_pos', help='sy_pos', type=float, default=1)
	parser.add_argument('--train_set', help='Percentage of the dataset used for training', type=float, default=train_set)
	args = parser.parse_args()

	return args


args = parse_args()

# [id, timestep (s), timestep (ns), pos x, pos y, yaw, vel x, vel y, omega, goal x, goal y]

pedestrian_data = np.genfromtxt(os.path.join(home + data_path, 'datafile.csv'), delimiter=",")[1:, :]

agent_container = ped_cont.AgentContainer()

for sample_idx in range(pedestrian_data.shape[0]):
    if pedestrian_data[sample_idx, 0] != -1:
        id = pedestrian_data[sample_idx, 0]
        timestamp = np.round(pedestrian_data[sample_idx, 1],1)# + pedestrian_data[sample_idx, 2] * 1e-9  # time in seconds
        pose = np.zeros([1,3])
        vel = np.zeros([1,3])
        pose[:,0:2] = np.true_divide(pedestrian_data[sample_idx, 3:5], np.array([norm_const_x, norm_const_y]))
        vel[:,0:2] = np.true_divide(pedestrian_data[sample_idx, 6:8], np.array([norm_const_vx, norm_const_vy]))
        #print(pedestrian_data[sample_idx, :])
        goal = np.true_divide(pedestrian_data[sample_idx, 9:], np.array([norm_const_x, norm_const_y]))

        agent_container.addDataSample(id, timestamp, pose, vel, goal)

dh = dhlstm.DataHandlerLSTM()

dh.load_map(args)