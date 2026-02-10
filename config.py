import os
import datetime as dt
import numpy as np
import itertools

# ==========================================
# 1. Action Space Configuration
# ==========================================
# Feasible PRB allocations (Slice 0, 1, 2)
# feasible_prb_allocation_all = [
#     [6, 39, 5], [12, 33, 6], [18, 27, 5], [24, 21, 5], [30, 15, 5], [36, 9, 5], [42, 3, 5],
#     [6, 33, 11], [12, 27, 11], [18, 21, 11], [24, 15, 11], [30, 9, 11], [36, 3, 11],
#     [6, 27, 17], [12, 21, 17], [18, 15, 17], [24, 6, 17], [30, 3, 17],
#     [6, 21, 23], [12, 15, 23], [18, 9, 23], [24, 3, 23],
#     [6, 15, 30], [12, 9, 30], [18, 3, 30],
#     [6, 9, 35], [12, 3, 35],
#     [6, 3, 41]
# ]

# # Using specific subset of indexes
# feasible_prb_allocation_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# scheduling_combos = [
#     [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2], [0,2,0], [0,2,1], [0,2,2],
#     [1,0,0], [1,0,1], [1,0,2], [1,1,0], [1,1,1], [1,1,2], [1,2,0], [1,2,1], [1,2,2],
#     [2,0,0], [2,0,1], [2,0,2], [2,1,0], [2,1,1], [2,1,2], [2,2,0], [2,2,1], [2,2,2]
# ]



feasible_prb_allocation_all = [
    [6, 39, 5], [6, 33, 11],[12, 33, 6], [36, 9, 5], [42, 3, 5],

]

# Using specific subset of indexes
feasible_prb_allocation_indexes = [0, 1, 2, 3, 4]

# scheduling_combos = [
#     [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2], [0,2,0], [0,2,1], [0,2,2],
#     [1,0,0], [1,0,1], [1,0,2], [1,1,0], [1,1,1], [1,1,2], [1,2,0], [1,2,1], [1,2,2],
#     [2,0,0], [2,0,1], [2,0,2], [2,1,0], [2,1,1], [2,1,2], [2,2,0], [2,2,1], [2,2,2]
# ]

scheduling_combos = [
    [0,0,0], [1,1,1], [2,2,2]
]



# Generate Action Space
actions = list(itertools.product(feasible_prb_allocation_indexes, scheduling_combos))
n_actions = len(actions)


metric_list_autoencoder = [
    "dl_buffer [bytes]", 
    "tx_brate downlink [Mbps]", 
    "tx_pkts downlink"
]

add_ratio_granted_req = True 

remove_zero_req_prb_entries = False
remove_zero_throughput_entries = True

# Data Paths
dataset_path = 'dataset/embb_dataset.csv' 
encoder_path = 'encoder.h5'

# ==========================================
# Environment Settings
# ==========================================
num_parallel_environments = 10
num_steps_per_episode = 10     
du_prb = 50
use_gpu_in_env = True

# ==========================================
# PPO Agent Hyperparameters
# ==========================================
ppo_actor_fc_layers = (5, 30) 
ppo_value_fc_layers = (5, 30)
ppo_learning_rate = 1e-3
ppo_num_epochs = 10
ppo_entropy_regularization = 0.1
ppo_importance_ratio_clipping = 0.2
ppo_normalize_observations = True
ppo_normalize_rewards = False
ppo_use_gae = True

# ==========================================
# Training Settings
# ==========================================
total_training_iterations = 100
collect_episodes_per_iteration = 100 
eval_interval = 25 
log_interval = 10  


drl_save_folder = './drl_agent_files'
checkpoint_dir = os.path.join(drl_save_folder, "checkpoints")
policy_dir = os.path.join(drl_save_folder, "policy")
log_dir = os.path.join(drl_save_folder, "logs", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))


run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
drl_save_folder = os.path.join('./drl_agent_files', f'run_{run_id}')

checkpoint_dir = os.path.join(drl_save_folder, "checkpoints")
policy_dir = os.path.join(drl_save_folder, "policy")
log_dir = os.path.join(drl_save_folder, "logs")