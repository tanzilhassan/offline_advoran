# dataset_builder.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


USELESS_COLUMNS = ['num_ues','slicing_enabled', 'power_multiplier', 'tx_errors downlink (%)', 'ul_rssi', 'dl_pmi', 'dl_ri', 'ul_n', 'dl_mcs', 'dl_n_samples', 'dl_cqi','ul_mcs','ul_n_samples', 'ul_buffer [bytes]','rx_brate uplink [Mbps]','rx_pkts uplink','rx_errors uplink (%)','ul_sinr','phr','ul_turbo_iters'] 

META_COLUMNS = ["Timestamp", "IMSI", "RNTI"]


# ----------------------------
# File discovery
# ----------------------------
def find_metrics_csvs(root_dir: str, suffix: str = "_metrics.csv") -> List[str]:
    """Generic + robust: recursively find all files ending with suffix."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    paths = sorted(str(p) for p in root.rglob(f"*{suffix}") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No '*{suffix}' files found under: {root_dir}")
    return paths


# ----------------------------
# Simple throughput-based reward for eMBB
# ----------------------------
def compute_reward_vectorized(
    thp_mbps: np.ndarray,
    slice_id: np.ndarray,
    scale: float = 100.0,
) -> np.ndarray:
    """
    Simple reward for eMBB slice based on throughput and slice_id.
    
    reward = throughput * scale  if slice_id == 0
    reward = 0                   if slice_id != 0
    
    Args:
        thp_mbps: Throughput array in Mbps
        slice_id: Slice ID array
        scale: Scaling factor (default 100.0)
    
    Returns:
        Reward array (NOT normalized, raw scaled values)
    """
    thp = np.asarray(thp_mbps, dtype=np.float32)
    thp = np.nan_to_num(thp, nan=0.0, posinf=0.0, neginf=0.0)
    
    slice_id = np.asarray(slice_id, dtype=np.int32)
    
    # Compute reward: throughput * scale for slice_id == 0, else 0
    reward = np.where(slice_id == 0, thp * scale, 0.0)
    
    return reward.astype(np.float32)


# ----------------------------
# Preprocess
# ----------------------------
def preprocess_dataset(
    df: pd.DataFrame,
    encoder_cols: Optional[List[str]] = None,
    add_prb_ratio: bool = True,
    replace_zero_with_one: bool = False,
    scale_autoencoder_cols: bool = True,
    scale_factor: float = 10.0,
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = True,
    ratio_col: str = "ratio_granted_req",
    requested_col: str = "sum_requested_prbs",
    granted_col: str = "sum_granted_prbs",
    drop_useless_and_meta: bool = True,
) -> Tuple[pd.DataFrame, Optional[MinMaxScaler]]:
    encoder_cols = encoder_cols or []

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")].copy()

    # Drop useless/meta columns only if present
    if drop_useless_and_meta:
        drop_cols = [c for c in (USELESS_COLUMNS + META_COLUMNS) if c in df.columns]
        if drop_cols:
            df = df.drop(drop_cols, axis=1)

    # Add PRB ratio (optional)
    if add_prb_ratio:
        if requested_col not in df.columns or granted_col not in df.columns:
            raise ValueError(f"Cannot add {ratio_col}: missing {requested_col} and/or {granted_col}")

        denom = pd.to_numeric(df[requested_col], errors="coerce").astype(np.float32).to_numpy()
        numer = pd.to_numeric(df[granted_col], errors="coerce").astype(np.float32).to_numpy()

        ratio = np.nan_to_num(numer / denom, nan=0.0, posinf=0.0, neginf=0.0)
        ratio = np.clip(ratio, 0.0, 1.0)

        if replace_zero_with_one:
            ratio = ratio.astype(np.float32, copy=False)
            ratio[denom <= 0] = 1.0

        df[ratio_col] = ratio.astype(np.float32, copy=False)

    # Scale ONLY the autoencoder input columns (MinMax -> *scale_factor)
    if scale_autoencoder_cols:
        if not encoder_cols:
            raise ValueError("scale_autoencoder_cols=True but encoder_cols is empty")
        missing = [c for c in encoder_cols if c not in df.columns]
        if missing:
            raise ValueError(f"encoder_cols missing in df: {missing}")

        X = df[encoder_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if scaler is None:
            scaler = MinMaxScaler()

        if fit_scaler:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)

        df.loc[:, encoder_cols] = (scale_factor * X_scaled).astype(np.float32)

    return df, scaler


# ----------------------------
# Streaming builder (memory-friendly)
# ----------------------------
def build_big_dataset_csv(
    root_dir: str,
    out_csv: str,
    suffix: str = "_metrics.csv",
    thp_col: str = "tx_brate downlink [Mbps]",
    slice_id_col: str = "slice_id",
    reward_col: str = "reward",
    reward_scale: float = 100.0,
    preprocess: bool = True,
    encoder_cols: Optional[List[str]] = None,
    scale_autoencoder_cols: bool = True,
    scale_factor: float = 10.0,
    add_prb_ratio: bool = True,
    replace_zero_with_one: bool = False,
    stream: bool = True,
    verbose: bool = True,
) -> None:
    """
    Build one big CSV from many *_metrics.csv files with simple throughput-based reward.
    
    Reward = throughput * reward_scale  if slice_id == 0
    Reward = 0                          if slice_id != 0
    
    NO NORMALIZATION - rewards are raw scaled throughput values.

    If stream=True:
      - Pass 1: fit MinMaxScaler for encoder columns (via partial_fit)
      - Pass 2: compute reward + preprocess, append to out_csv
    """
    encoder_cols = encoder_cols or []

    csv_paths = find_metrics_csvs(root_dir, suffix=suffix)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # avoid accidental append to stale file

    if not stream:
        # Simple in-memory concat (can blow RAM for very large datasets)
        dfs = []
        it = tqdm(csv_paths, desc="Reading CSVs", unit="file") if verbose else csv_paths
        for p in it:
            dfs.append(pd.read_csv(p))
        df = pd.concat(dfs, ignore_index=True)

        if thp_col not in df.columns:
            raise ValueError(f"Missing throughput column '{thp_col}' in concatenated df")
        if slice_id_col not in df.columns:
            raise ValueError(f"Missing slice_id column '{slice_id_col}' in concatenated df")

        # Compute reward (NO normalization)
        df[reward_col] = compute_reward_vectorized(
            df[thp_col].to_numpy(),
            df[slice_id_col].to_numpy(),
            scale=reward_scale,
        )

        scaler = None
        if preprocess:
            df, scaler = preprocess_dataset(
                df,
                encoder_cols=encoder_cols,
                add_prb_ratio=add_prb_ratio,
                replace_zero_with_one=replace_zero_with_one,
                scale_autoencoder_cols=scale_autoencoder_cols,
                scale_factor=scale_factor,
                scaler=None,
                fit_scaler=True,
            )

        df.to_csv(out_csv, index=False)
        if verbose:
            print(f"[OK] Wrote big dataset: {out_csv}  (rows={len(df)})")
            print(f"     Reward = throughput × {reward_scale} (slice_id == 0 only, NO normalization)")
        return

    # ---------------- Pass 1: scaler fit only ----------------
    scaler = MinMaxScaler() if (preprocess and scale_autoencoder_cols) else None
    needs_scaler = scaler is not None

    it1 = tqdm(csv_paths, desc="Pass 1/2 (scaler fit)", unit="file") if verbose else csv_paths
    for p in it1:
        df1 = pd.read_csv(p)
        
        # Check required columns
        if thp_col not in df1.columns:
            raise ValueError(f"Missing '{thp_col}' in file: {p}")
        if slice_id_col not in df1.columns:
            raise ValueError(f"Missing '{slice_id_col}' in file: {p}")

        if needs_scaler:
            missing = [c for c in encoder_cols if c not in df1.columns]
            if missing:
                raise ValueError(f"encoder_cols missing in file {p}: {missing}")

            X = df1[encoder_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            scaler.partial_fit(X)

    # ---------------- Pass 2: write output incrementally ----------------
    first_write = True
    it2 = tqdm(csv_paths, desc="Pass 2/2 (write)", unit="file") if verbose else csv_paths
    for p in it2:
        df2 = pd.read_csv(p)

        # Compute reward (NO normalization)
        df2[reward_col] = compute_reward_vectorized(
            df2[thp_col].to_numpy(),
            df2[slice_id_col].to_numpy(),
            scale=reward_scale,
        )

        if preprocess:
            df2, _ = preprocess_dataset(
                df2,
                encoder_cols=encoder_cols,
                add_prb_ratio=add_prb_ratio,
                replace_zero_with_one=replace_zero_with_one,
                scale_autoencoder_cols=scale_autoencoder_cols,
                scale_factor=scale_factor,
                scaler=scaler,
                fit_scaler=False,  # IMPORTANT: use pass-1 fitted scaler
            )

        df2.to_csv(out_csv, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False

    if verbose:
        print(f"[OK] Wrote big dataset: {out_csv}")
        print(f"     Reward = throughput × {reward_scale} (slice_id == 0 only, NO normalization)")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build eMBB dataset with throughput-based rewards (slice_id == 0 only)"
    )
    parser.add_argument("--root", type=str, required=True, 
                        help="Root directory to scan recursively")
    parser.add_argument("--out", type=str, default="datasets/embb_dataset.csv", 
                        help="Output CSV path")
    parser.add_argument("--suffix", type=str, default="_metrics.csv", 
                        help="CSV filename suffix")
    parser.add_argument("--thp_col", type=str, default="tx_brate downlink [Mbps]", 
                        help="Throughput column name")
    parser.add_argument("--slice_id_col", type=str, default="slice_id",
                        help="Slice ID column name")
    parser.add_argument("--reward_scale", type=float, default=100.0,
                        help="Scaling factor for reward (reward = throughput * scale)")
    
    # Preprocessing options
    parser.add_argument("--no_preprocess", action="store_true", 
                        help="Disable preprocessing")
    parser.add_argument("--stream", action="store_true", 
                        help="Use streaming 2-pass build (recommended for big data)")

    # Encoder scaling args
    parser.add_argument(
        "--encoder_cols",
        nargs="*",
        default=["dl_buffer [bytes]", "tx_brate downlink [Mbps]", "tx_pkts downlink"],
        help="Columns to scale (MinMax then *scale_factor)",
    )
    parser.add_argument("--no_scale_encoder", action="store_true", 
                        help="Disable scaling encoder cols")
    parser.add_argument("--scale_factor", type=float, default=10.0, 
                        help="Multiply scaled encoder cols by this factor")

    # PRB ratio args
    parser.add_argument("--no_prb_ratio", action="store_true", 
                        help="Disable adding ratio_granted_req column")
    parser.add_argument("--replace_zero_with_one", action="store_true", 
                        help="If requested_prbs<=0, set ratio=1.0")

    args = parser.parse_args()

    build_big_dataset_csv(
        root_dir=args.root,
        out_csv=args.out,
        suffix=args.suffix,
        thp_col=args.thp_col,
        slice_id_col=args.slice_id_col,
        reward_scale=args.reward_scale,
        preprocess=(not args.no_preprocess),
        encoder_cols=args.encoder_cols,
        scale_autoencoder_cols=(not args.no_scale_encoder),
        scale_factor=args.scale_factor,
        add_prb_ratio=(not args.no_prb_ratio),
        replace_zero_with_one=args.replace_zero_with_one,
        stream=args.stream,
        verbose=True,
    )