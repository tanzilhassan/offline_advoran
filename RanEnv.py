import logging
from typing import Dict, Any, Optional, Tuple, List
from itertools import product
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from tensorflow import keras

# ==========================================
# Constants
# ==========================================

SLICE_NAMES = {0: 'eMBB', 1: 'MTC', 2: 'URLLC'}

# Columns configuration
COLUMNS_ENCODER = ["dl_buffer [bytes]", "tx_brate downlink [Mbps]", "tx_pkts downlink"]
COLUMNS_REWARD = "reward"

# The specific subset of PRB allocations used (Slice 0, 1, 2)
FEASIBLE_PRB_ALLOCATIONS = [
    [6, 39, 5], [12, 33, 6], [18, 27, 5], [24, 21, 5], [30, 15, 5], [36, 9, 5], [42, 3, 5],
    [6, 33, 11], [12, 27, 11], [18, 21, 11], [24, 15, 11], [30, 9, 11], [36, 3, 11]
]

SCHEDULING_COMBOS = [
    [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2], [0,2,0], [0,2,1], [0,2,2],
    [1,0,0], [1,0,1], [1,0,2], [1,1,0], [1,1,1], [1,1,2], [1,2,0], [1,2,1], [1,2,2],
    [2,0,0], [2,0,1], [2,0,2], [2,1,0], [2,1,1], [2,1,2], [2,2,0], [2,2,1], [2,2,2]
]

class RanMultiSliceEnv(gym.Env):
    """
    Optimized RAN Environment for TF-Agents.
    Implements the 'Old Gym API' (4-tuple return) for direct compatibility.
    """
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
        
        # Validation
        if data is None: raise ValueError("Dataframe required.")
        self.df = data
        self.max_steps = max_steps
        self.n_samples_per_slice = n_samples_per_slice
        self.du_prb = du_prb
        self.use_mean_obs = use_mean_obs
        self.num_slices = 3
        
        # Logging setup
        self.logger = logging.getLogger("RanEnv")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        # 1. Setup RBG Size
        self.rbg_size = self._get_rbg_size(du_prb)

        # 2. Setup Action Space & Pre-computation
        # Combine PRB and Scheduling into a single list of all possible actions
        all_combinations = list(product(FEASIBLE_PRB_ALLOCATIONS, SCHEDULING_COMBOS))
        
        # Pre-compute data indices map: {slice_id: {(prb, sched): [row_indices]}}
        self.slice_indices = self._map_data_indices()
        
        # Filter valid actions (keep only those with data)
        self.valid_actions = []
        for i, (prb_alloc, sched_alloc) in enumerate(all_combinations):
            if self._check_data_availability(prb_alloc, sched_alloc):
                self.valid_actions.append((prb_alloc, sched_alloc))

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.logger.info(f"Initialized with {len(self.valid_actions)} valid actions.")

        # 3. Setup Encoder & Observation Space
        self.encoder = None
        encoder_out_dim = 3 # Default if no encoder
        
        if encoder_path:
            try:
                self.encoder = keras.models.load_model(encoder_path, compile=False)
                # Infer output dim from a dummy pass
                dummy = np.zeros((1, n_samples_per_slice, len(COLUMNS_ENCODER)), dtype=np.float32)
                encoder_out_dim = self.encoder.predict(dummy, verbose=0).shape[-1]
            except Exception as e:
                self.logger.warning(f"Could not load encoder: {e}. Using mean features.")

        # Obs: (Encoder_Out * 3) + (PRB_Alloc * 3) + (Sched_Alloc * 3)
        obs_dim = (encoder_out_dim * 3) + 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # State tracking
        self.current_step = 0
        self.current_config = None # Tuple(prb, sched)

    def _get_rbg_size(self, prb_count: int) -> int:
        if prb_count <= 10: return 1
        if prb_count <= 26: return 2
        if prb_count <= 63: return 3
        return 4

    def _map_data_indices(self) -> Dict:
        """Optimized index mapping for O(1) lookups."""
        mapping = {i: {} for i in range(self.num_slices)}
        for slice_id in range(self.num_slices):
            sdf = self.df[self.df['slice_id'] == slice_id]
            # Group by configuration for fast access
            grouped = sdf.groupby(['slice_prb', 'scheduling_policy']).indices
            mapping[slice_id] = grouped
        return mapping

    def _check_data_availability(self, prb_alloc, sched_alloc) -> bool:
        """Returns True if data exists for all 3 slices in this config."""
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
            
            # Get Data
            if len(indices) == 0:
                # Fallback: Zero padding
                data_chunk = np.zeros((self.n_samples_per_slice, len(COLUMNS_ENCODER)), dtype=np.float32)
            elif self.use_mean_obs:
                # Deterministic mean (Faster)
                mean_vals = self.df.loc[indices, COLUMNS_ENCODER].mean().values
                data_chunk = np.tile(mean_vals, (self.n_samples_per_slice, 1)).astype(np.float32)
            else:
                # Random sampling
                sample_idx = np.random.choice(indices, self.n_samples_per_slice, replace=(len(indices) < self.n_samples_per_slice))
                data_chunk = self.df.loc[sample_idx, COLUMNS_ENCODER].values.astype(np.float32)

            # Scaling
            data_chunk[:, 0] /= 100000.0 # Scale dl_buffer

            # Encode
            if self.encoder:
                # Warning: .predict is slow inside a loop. Consider pre-computing if possible.
                batch = np.expand_dims(data_chunk, 0)
                encoded = self.encoder.predict(batch, verbose=0).squeeze()
            else:
                # Simple average if no encoder
                encoded = np.mean(data_chunk, axis=0) # Reduces to 3D
            
            features.append(encoded)

        # Flatten and concatenate all parts
        obs = np.concatenate([
            np.concatenate(features),
            np.array(prb_alloc, dtype=np.float32),
            np.array(sched_alloc, dtype=np.float32)
        ], dtype=np.float32)
        
        return obs

    def step(self, action: int):
        """
        Modified to return 4 values for TF-Agents compatibility:
        (observation, reward, done, info)
        """
        self.current_step += 1
        prb_alloc, sched_alloc = self.valid_actions[action]
        self.current_config = (prb_alloc, sched_alloc)

        # 1. Calculate Reward (Slice 0 only)
        key_s0 = (prb_alloc[0], sched_alloc[0])
        idx_s0 = self.slice_indices[0].get(key_s0, [])
        
        if len(idx_s0) > 0:
            reward = self.df.loc[idx_s0, COLUMNS_REWARD].mean()
        else:
            reward = -1.0 # Penalty for missing data

        # 2. Get Observation
        obs = self._get_observation(prb_alloc, sched_alloc)
        
        # 3. Check Done
        done = self.current_step >= self.max_steps
        
        # 4. Info
        info = {
            "prb": prb_alloc, 
            "sched": sched_alloc,
            "rbg": [math.ceil(p / self.rbg_size) for p in prb_alloc]
        }

        # Cast types strictly for TF
        return obs, np.float32(reward), done, info

    def reset(self, seed=None, options=None):
        """
        Modified to return 1 value for TF-Agents compatibility:
        observation
        """
        super().reset(seed=seed)
        self.current_step = 0
        
        # Pick random initial valid action
        rand_idx = np.random.randint(len(self.valid_actions))
        prb_alloc, sched_alloc = self.valid_actions[rand_idx]
        self.current_config = (prb_alloc, sched_alloc)
        
        obs = self._get_observation(prb_alloc, sched_alloc)
        return obs # Return only obs (No info dict)

    def render(self, mode="ansi"):
        if mode == "ansi" and self.current_config:
            prb, sched = self.current_config
            rbg = [math.ceil(p / self.rbg_size) for p in prb]
            print(f"Step: {self.current_step} | PRB: {prb} | RBG: {rbg} | Sched: {sched}")




if __name__ == "__main__":
    import os

    # ==========================================
    # 1. Configuration
    # ==========================================
    CSV_PATH = '../dataset/dataset.csv'  # <--- Make sure this path is correct
    ENCODER_PATH = 'encoder.h5'       # <--- Optional: Set to None if you don't have it yet

    print(f"Checking for dataset at: {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: File not found at {CSV_PATH}")
        print("Please ensure the file is in the correct directory.")
    else:
        # ==========================================
        # 2. Load Real Dataset
        # ==========================================
        print("Loading dataset...")
        # Load only necessary columns to save memory
        required_cols = ['slice_id', 'slice_prb', 'scheduling_policy', 'reward'] + COLUMNS_ENCODER
        
        df = pd.read_csv(CSV_PATH, usecols=required_cols)
        
        # Optimize types to save memory
        df['slice_id'] = df['slice_id'].astype(np.int16)
        df['slice_prb'] = df['slice_prb'].astype(np.int16)
        df['scheduling_policy'] = df['scheduling_policy'].astype(np.int8)
        df['reward'] = df['reward'].astype(np.float32)
        for col in COLUMNS_ENCODER:
            df[col] = df[col].astype(np.float32)
            
        print(f" Dataset loaded: {len(df)} rows")

        # ==========================================
        # 3. Initialize Environment
        # ==========================================
        print("\nInitializing Environment...")
        env = RanMultiSliceEnv(
            data=df,
            encoder_path=ENCODER_PATH if os.path.exists(str(ENCODER_PATH)) else None,
            max_steps=20,  # Run a short episode
            du_prb=50
        )

        # ==========================================
        # 4. Test Reset
        # ==========================================
        print("\nTesting Reset...")
        initial_obs = env.reset()
        print(f"Observation Shape: {initial_obs.shape}")
        
        # ==========================================
        # 5. Run a Test Episode
        # ==========================================
        print("\nRunning Test Episode...")
        total_reward = 0.0
        
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"Step {i+1}: Reward {reward:.4f} | PRB {info['prb']} | Sched {info['sched']}")
            
            if done:
                break
                
        print(f"\n✓ Test Complete. Total Reward: {total_reward:.4f}")