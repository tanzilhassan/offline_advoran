import logging
import math
from itertools import product
from typing import Dict, Any, Optional

import gym
from gym import spaces
import numpy as np
import pandas as pd
from tensorflow import keras
import config  # Import local config

class RanMultiSliceEnv(gym.Env):
    metadata = {'render_modes': ['ansi']}

    def __init__(
        self, 
        data: pd.DataFrame, 
        encoder_path: Optional[str] = None,
        max_steps: int = 100,
        n_samples_per_slice: int = 10,
        du_prb: int = 50,
        use_mean_obs: bool = True
    ):
        super().__init__()
        
        if data is None: raise ValueError("Dataframe required.")
        
        # ------------------------------------------
        # 1. Data Filtering & Feature Engineering
        # ------------------------------------------
        self.df = data.copy() 
        
        # A. Feature Engineering: Ratio
        if config.add_ratio_granted_req:
            # Avoid division by zero with 1e-9
            self.df['ratio_granted_req'] = self.df['sum_granted_prbs'] / (self.df['sum_requested_prbs'] + 1e-9)
            self.df['ratio_granted_req'] = self.df['ratio_granted_req'].clip(0, 1)

        # B. Filter: Zero Throughput
        if config.remove_zero_throughput_entries:
            initial_len = len(self.df)
            # Assuming 'tx_brate downlink [Mbps]' is the metric
            self.df = self.df[self.df['tx_brate downlink [Mbps]'] > 0]
            # print(f"Env Init: Filtered Zero Throughput ({initial_len} -> {len(self.df)} rows)")

        # C. Filter: Zero Requested PRBs
        if config.remove_zero_req_prb_entries:
            initial_len = len(self.df)
            self.df = self.df[self.df['sum_requested_prbs'] > 0]
            # print(f"Env Init: Filtered Zero Req PRBs ({initial_len} -> {len(self.df)} rows)")
            
        # ------------------------------------------
        # 2. Standard Initialization
        # ------------------------------------------
        self.max_steps = max_steps
        self.n_samples_per_slice = n_samples_per_slice
        self.du_prb = du_prb
        self.use_mean_obs = use_mean_obs
        self.num_slices = 3
        self.columns_encoder = config.metric_list_autoencoder
        
        self.logger = logging.getLogger("RanEnv")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        self.rbg_size = self._get_rbg_size(du_prb)
        
        # Generate all combinations
        all_combinations = list(product(config.feasible_prb_allocation_all, config.scheduling_combos))
        
        # Map data indices (Optimized)
        self.slice_indices = self._map_data_indices()
        
        # Determine Valid Actions (Intersection of Config Indexes AND Data Availability)
        self.valid_actions = []
        for i, (prb_alloc, sched_alloc) in enumerate(all_combinations):
            if i in config.feasible_prb_allocation_indexes: 
                if self._check_data_availability(prb_alloc, sched_alloc):
                    self.valid_actions.append((prb_alloc, sched_alloc))

        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        # Encoder Setup
        self.encoder = None
        encoder_out_dim = len(self.columns_encoder) 
        
        if encoder_path:
            try:
                self.encoder = keras.models.load_model(encoder_path, compile=False)
                # Dummy pass to get output shape
                dummy = np.zeros((1, n_samples_per_slice, len(self.columns_encoder)), dtype=np.float32)
                encoder_out_dim = self.encoder.predict(dummy, verbose=0).shape[-1]
            except Exception as e:
                self.logger.warning(f"Encoder load error: {e}. Using raw mean features.")

        # Observation Space: (Features*3) + (PRB*3) + (Sched*3)
        obs_dim = (encoder_out_dim * 3) + 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.current_step = 0
        self.current_config = None

    def _get_rbg_size(self, prb_count: int) -> int:
        if prb_count <= 10: return 1
        if prb_count <= 26: return 2
        if prb_count <= 63: return 3
        return 4

    def _map_data_indices(self) -> Dict:
        mapping = {i: {} for i in range(self.num_slices)}
        for slice_id in range(self.num_slices):
            sdf = self.df[self.df['slice_id'] == slice_id]
            grouped = sdf.groupby(['slice_prb', 'scheduling_policy']).indices
            mapping[slice_id] = grouped
        return mapping

    def _check_data_availability(self, prb_alloc, sched_alloc) -> bool:
        for s_id in range(self.num_slices):
            key = (prb_alloc[s_id], sched_alloc[s_id])
            if key not in self.slice_indices[s_id]:
                return False
        return True

    def _get_observation(self, prb_alloc, sched_alloc) -> np.ndarray:
        features = []
        
        for s_id in range(self.num_slices):
            key = (prb_alloc[s_id], sched_alloc[s_id])
            indices = self.slice_indices[s_id].get(key, [])
            
            # Retrieve Data
            if len(indices) == 0:
                data_chunk = np.zeros((self.n_samples_per_slice, len(self.columns_encoder)), dtype=np.float32)
            elif self.use_mean_obs:
                mean_vals = self.df.iloc[indices][self.columns_encoder].mean().values
                data_chunk = np.tile(mean_vals, (self.n_samples_per_slice, 1)).astype(np.float32)
            else:
                sample_idx = np.random.choice(indices, self.n_samples_per_slice, replace=(len(indices) < self.n_samples_per_slice))
                data_chunk = self.df.iloc[sample_idx][self.columns_encoder].values.astype(np.float32)

            # Specific Scaling for 'dl_buffer'
            if "dl_buffer [bytes]" in self.columns_encoder:
                idx = self.columns_encoder.index("dl_buffer [bytes]")
                data_chunk[:, idx] /= 100000.0

            # Encode
            if self.encoder:
                batch = np.expand_dims(data_chunk, 0)
                encoded = self.encoder.predict(batch, verbose=0).reshape(-1)
            else:
                encoded = np.mean(data_chunk, axis=0) 
            
            features.append(encoded)

        # Concatenate: [Features(9D), PRB(3D), Sched(3D)]
        obs = np.concatenate([
            np.concatenate(features),
            np.array(prb_alloc, dtype=np.float32),
            np.array(sched_alloc, dtype=np.float32)
        ], dtype=np.float32)
        
        return obs

    def step(self, action: int):
        self.current_step += 1
        prb_alloc, sched_alloc = self.valid_actions[action]
        self.current_config = (prb_alloc, sched_alloc)

        # Reward (Slice 0 eMBB only)
        key_s0 = (prb_alloc[0], sched_alloc[0])
        idx_s0 = self.slice_indices[0].get(key_s0, [])
        
        if len(idx_s0) > 0:
            reward = self.df.iloc[idx_s0]['reward'].mean()
        else:
            reward = -1.0

        obs = self._get_observation(prb_alloc, sched_alloc)
        done = self.current_step >= self.max_steps
        
        info = {
            "prb": prb_alloc, 
            "sched": sched_alloc,
            "rbg": [math.ceil(p / self.rbg_size) for p in prb_alloc]
        }

        # TF-Agents requires 4 values from step
        return obs, np.float32(reward), done, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        rand_idx = np.random.randint(len(self.valid_actions))
        prb_alloc, sched_alloc = self.valid_actions[rand_idx]
        self.current_config = (prb_alloc, sched_alloc)
        obs = self._get_observation(prb_alloc, sched_alloc)
        return obs 

    def render(self, mode="ansi"):
        if mode == "ansi" and self.current_config:
            prb, sched = self.current_config
            print(f"Step: {self.current_step} | PRB: {prb} | Sched: {sched}")