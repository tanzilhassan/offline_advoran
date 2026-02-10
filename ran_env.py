import logging
import math
from itertools import product
from typing import Dict, Any, Optional

import os
import gym
from gym import spaces
import numpy as np
import pandas as pd
from tensorflow import keras
import config  


class RanEnv(gym.Env):
    metadata = {'render_modes': ['ansi']}

    def __init__(
        self, 
        data_bundle: dict, 
        encoder_path: Optional[str] = None,
        max_steps: int = 10,
        n_samples_per_slice: int = 10,
        du_prb: int = 50,
        use_mean_obs: bool = True
    ):
        super().__init__()
        
        self.logger = logging.getLogger("RanEnv")
        
        # Unpack Bundle
        self.data_cache = data_bundle['data']
        self.num_metrics = data_bundle.get('num_metrics', len(config.metric_list_autoencoder))
        
        # Config
        self.max_steps = max_steps
        self.n_samples_per_slice = n_samples_per_slice
        self.du_prb = du_prb
        self.use_mean_obs = use_mean_obs
        self.num_slices = 3
        self.columns_encoder = config.metric_list_autoencoder
        

        self.all_combinations = list(product(config.feasible_prb_allocation_all, config.scheduling_combos))
        

        self.valid_actions = [self.all_combinations[i] for i in range(len(self.all_combinations))]

        self.action_space = spaces.Discrete(len(self.valid_actions))
        

        self.encoder = None
        encoder_out_dim = len(self.columns_encoder) 
        if encoder_path and os.path.exists(encoder_path):
            try:
                self.encoder = keras.models.load_model(encoder_path, compile=False)
                dummy = np.zeros((1, n_samples_per_slice, len(self.columns_encoder)), dtype=np.float32)
                encoder_out_dim = self.encoder.predict(dummy, verbose=0).shape[-1]
            except Exception as e:
                self.logger.warning(f"Encoder load failed: {e}. Using raw mean.")

        per_slice_dim = encoder_out_dim + 1 + 1 
        total_obs_dim = per_slice_dim * 3
        
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(total_obs_dim,), dtype=np.float32)
        
        self.current_step = 0
        self.current_config = None
        self.last_reward = 0.0

    def _get_observation_and_reward(self, prb_alloc, sched_alloc):
        slice_obs_list = []
        total_reward = 0.0
        
        for s_id in range(self.num_slices):
            key = (prb_alloc[s_id], sched_alloc[s_id])
            
            if key in self.data_cache[s_id]:
                all_data = self.data_cache[s_id][key]
                count = len(all_data)
                
                indices = np.random.choice(count, self.n_samples_per_slice, replace=(count < self.n_samples_per_slice))
                data_chunk = all_data[indices]
                
                metrics_chunk = data_chunk[:, :self.num_metrics]
                
                if s_id == 0:
                    total_reward = np.mean(data_chunk[:, -1])
            else:
                metrics_chunk = np.zeros((self.n_samples_per_slice, self.num_metrics), dtype=np.float32)
                if s_id == 0: 
                    total_reward = -1.0 

            # Encode Metrics
            if self.encoder:
                batch = np.expand_dims(metrics_chunk, 0)
                encoded = self.encoder.predict(batch, verbose=0).reshape(-1)
            else:
                encoded = np.mean(metrics_chunk, axis=0)
            
            # Construct Slice Vector [Encoded, PRB, Sched]
            slice_vec = np.concatenate([
                encoded,
                [float(prb_alloc[s_id])], 
                [float(sched_alloc[s_id])]
            ])
            slice_obs_list.append(slice_vec)

        final_obs = np.concatenate(slice_obs_list).astype(np.float32)
        return final_obs, float(total_reward)

    def step(self, action_idx):
        self.current_step += 1
        prb, sched = self.valid_actions[action_idx]
        self.current_config = (prb, sched)

        obs, reward = self._get_observation_and_reward(prb, sched)
        self.last_reward = reward
        
        done = self.current_step >= self.max_steps
        info = {"prb": prb, "sched": sched}

        return obs, np.float32(reward), done, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_reward = 0.0
        
        rand_idx = np.random.randint(len(self.valid_actions))
        prb, sched = self.valid_actions[rand_idx]
        self.current_config = (prb, sched)
        
        obs, _ = self._get_observation_and_reward(prb, sched)
        return obs

    def render(self, mode="ansi"):
        if self.current_config:
            prb, sched = self.current_config
            print(f"Step: {self.current_step:2d} | Action: PRB {prb}, Sched {sched} | Reward: {self.last_reward:.4f}")

# ==========================================
# Main Part: Data Loading and Testing
# ==========================================
if __name__ == "__main__":
    import os
    np.random.seed(42) # For reproducibility
    
    print("--- Starting RAN Environment Test ---")
    
    # Load Dataset
    csv_path = config.dataset_path

    df = pd.read_csv(csv_path)
    

    feature_cache = {0: {}, 1: {}, 2: {}}
    metric_cols = config.metric_list_autoencoder
    context_cols = ['slice_id', 'slice_prb', 'scheduling_policy']
    
    if 'reward' not in df.columns:
        df['reward'] = 0.0 
        
    final_feature_order = metric_cols + context_cols + ['reward']
    
    for s_id in range(3):
        slice_df = df[df['slice_id'] == s_id]
        grouped = slice_df.groupby(['slice_prb', 'scheduling_policy'])
        
        for (prb, sched), group in grouped:
            feats = group[final_feature_order].values.astype(np.float32)
            feature_cache[s_id][(prb, sched)] = feats
            
    dummy_bundle = {
        'data': feature_cache,
        'num_metrics': len(metric_cols)
    }


    env = RanEnv(
        data_bundle=dummy_bundle,
        encoder_path=config.encoder_path if os.path.exists(config.encoder_path) else None,
        n_samples_per_slice=10,
        max_steps=config.num_steps_per_episode
    )
    
    print(f"Action Space Size: {env.action_space.n} (Expect 18)")
    print(f"Observation Space Shape: {env.observation_space.shape}")


    print("\n--- Running Sample Episode ---")
    obs = env.reset()
    for i in range(5):
        action_idx = env.action_space.sample() 
        obs, reward, done, info = env.step(action_idx)
        
        env.render() 
        
        if done:
            break
            
    print("\nTest Complete.")