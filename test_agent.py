import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tf_agents.environments import tf_py_environment
from tf_agents.environments import gym_wrapper 
from tf_agents.trajectories import time_step as ts

from ran_env import RanEnv
import config

def create_single_test_env():
    print("Loading Dataset for Test Env...")
    csv_path = 'dataset/embb_dataset.csv' 
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    feature_cache = {0: {}, 1: {}, 2: {}}
    metric_cols = config.metric_list_autoencoder
    context_cols = ['slice_id', 'slice_prb', 'scheduling_policy']
    
    if 'reward' not in df.columns: df['reward'] = 0.0
    final_feature_order = metric_cols + context_cols + ['reward']
    
    for s_id in range(3):
        slice_df = df[df['slice_id'] == s_id]
        grouped = slice_df.groupby(['slice_prb', 'scheduling_policy'])
        for (prb, sched), group in grouped:
            feats = group[final_feature_order].values.astype(np.float32)
            feature_cache[s_id][(prb, sched)] = feats

    data_bundle = {'data': feature_cache, 'num_metrics': len(metric_cols)}

    gym_env = RanEnv(
        data_bundle=data_bundle,
        n_samples_per_slice=10,
        max_steps=10 
    )

    # Wrap  
    py_env = gym_wrapper.GymWrapper(gym_env)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    
    return tf_env

def main():
    policy_path = 'drl_agent_files/run_20260209_092523/policy'
    print(f"--- Loading Policy from: {policy_path} ---")

    try:
        loaded_policy = tf.saved_model.load(policy_path)
        print("Policy loaded successfully.")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    tf_env = create_single_test_env()
    

    py_env_instance = tf_env.pyenv.envs[0].gym

    num_test_episodes = 3
    
    for i in range(num_test_episodes):
        print(f"\n--- Test Episode {i+1} ---")
        time_step = tf_env.reset()
        episode_reward = 0.0
        step_count = 0
        
        while not time_step.is_last():
            step_count += 1
            
            # Get Action from Policy
            action_step = loaded_policy.action(time_step)
            action_idx = action_step.action.numpy()[0]

            # Decode the action using the referenced python environment
            prb_alloc, sched_alloc = py_env_instance.valid_actions[action_idx]

            time_step = tf_env.step(action_step.action)
            reward = time_step.reward.numpy()[0]
            episode_reward += reward
            
            print(f"Step {step_count}:")
            print(f"  Action Idx: {action_idx}")
            print(f"  PRB Alloc : {prb_alloc}")  
            print(f"  Sched Pol : {sched_alloc}") 
            print(f"  Reward    : {reward:.4f}")
            print("-" * 30)

        print(f"Episode Finished. Total Reward: {episode_reward:.4f}")

if __name__ == "__main__":
    main()