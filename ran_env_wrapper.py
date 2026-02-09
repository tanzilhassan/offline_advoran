import pandas as pd
import numpy as np
import tensorflow as tf
import os

from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import gym_wrapper
from ran_env import RanMultiSliceEnv
import config

# Global Dataframe (Loads once to share memory)
global_df = None

def load_global_data():
    """Loads dataset from CSV into global memory if not already loaded."""
    global global_df
    
    # Return immediately if already loaded
    if global_df is not None:
        return global_df

    # Ensure we are looking for a CSV, even if config says .parquet
    csv_path = config.dataset_path.replace('.parquet', '.csv')
    
    print(f"Wrapper: Loading CSV from {csv_path}...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    # Load the CSV
    # Note: If memory is an issue, we can add 'usecols' here to limit columns
    global_df = pd.read_csv(csv_path)

    # Basic type optimization to reduce RAM usage (Crucial for 2GB CSVs)
    # slice_id is categorical, int16 is safer/smaller than float
    global_df['slice_id'] = global_df['slice_id'].astype(np.int16) 
    global_df['reward'] = global_df['reward'].astype(np.float32)
    
    # Ensure other columns used by the env are floats
    for col in config.metric_list_autoencoder:
        if col in global_df.columns:
            global_df[col] = global_df[col].astype(np.float32)

    return global_df

def create_py_env():
    """Factory function to create a single Python environment instance."""
    # Ensure data is loaded
    df = load_global_data()
    
    # Create the base Python environment
    return RanMultiSliceEnv(
        data=df,
        encoder_path=config.encoder_path if os.path.exists(config.encoder_path) else None,
        max_steps=config.num_steps_per_episode,
        du_prb=config.du_prb
    )

def get_training_env():
    """Creates a Single TF Environment for training."""
    load_global_data() 
    print("Wrapper: Creating Single Training Environment...")
    
    # 1. Create Raw Gym Env
    gym_env = create_py_env()
    
    # 2. Wrap it to convert to PyEnvironment (The missing link!)
    py_env = gym_wrapper.GymWrapper(gym_env)
    
    # 3. Wrap it to convert to TensorFlow Graph Env
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    
    return tf_env

def get_eval_env():
    """Creates a single TF Environment for evaluation."""
    load_global_data()
    print("Wrapper: Creating Single Evaluation Environment...")
    
    gym_env = create_py_env()
    
    # Wrap it here too
    py_env = gym_wrapper.GymWrapper(gym_env)
    
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    return tf_env