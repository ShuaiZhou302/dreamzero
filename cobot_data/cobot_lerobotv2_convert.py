#!/usr/bin/env python3
"""
Convert ALOHA-style HDF5 episodes to a minimal LeRobot v2 dataset layout.

Input (example):
  /path/to/cooking2/episode_0.hdf5
  /path/to/cooking2/episode_1.hdf5

Output (example):
  cobot_dataset/
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.cam_high/episode_000000.mp4
    videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
    videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4
    meta/info.json

This script writes a "minimal usable" LeRobot v2 format for DreamZero step-1
(`scripts/data/convert_lerobot_to_gear.py`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    import h5py
except Exception as e:  # pragma: no cover - import-time environment issue
    raise RuntimeError(
        "Failed to import h5py. Please use an environment with compatible "
        "numpy/h5py versions."
    ) from e


CAMERA_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CHUNK_SIZE = 1000


def parse_episode_idx(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.hdf5$", path.name)
    if not match:
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(match.group(1))


def get_hdf5_paths(input_dir: Path) -> List[Path]:
    paths = sorted(input_dir.glob("episode_*.hdf5"), key=parse_episode_idx)
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {input_dir}")
    return paths


def ensure_output_layout(output_dir: Path) -> None:
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)


def write_video_mp4(frames: np.ndarray, output_path: Path, fps: float) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shape [T,H,W,3], got {frames.shape}")

    height, width = int(frames.shape[1]), int(frames.shape[2])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    try:
        for frame in frames:
            # HDF5 images are RGB; OpenCV expects BGR.
            writer.write(frame[:, :, ::-1])
    finally:
        writer.release()


def build_episode_dataframe(
    qpos: np.ndarray,
    action: np.ndarray,
    episode_idx: int,
    fps: float,
) -> pd.DataFrame:
    if len(qpos) != len(action):
        raise ValueError(
            f"qpos/action length mismatch in episode {episode_idx}: "
            f"{len(qpos)} vs {len(action)}"
        )

    t = len(qpos)
    timestamps = np.arange(t, dtype=np.float64) / float(fps)
    # LeRobot parquet stores vector features as per-row arrays.
    df = pd.DataFrame(
        {
            "observation.state": [row.astype(np.float32) for row in qpos],
            "action": [row.astype(np.float32) for row in action],
            "timestamp": timestamps,
            "frame_index": np.arange(t, dtype=np.int64),
            "episode_index": np.full((t,), episode_idx, dtype=np.int64),
            "index": np.arange(t, dtype=np.int64),
            "task_index": np.zeros((t,), dtype=np.int64),
            # Keep an empty language field so downstream task extraction works.
            "annotation.task": [""] * t,
        }
    )
    return df


def convert_episode(
    h5_path: Path,
    out_root: Path,
    out_episode_idx: int,
    fps: float,
) -> Tuple[int, int, int]:
    chunk_idx = out_episode_idx // CHUNK_SIZE
    chunk_name = f"chunk-{chunk_idx:03d}"
    parquet_name = f"episode_{out_episode_idx:06d}.parquet"
    video_name = f"episode_{out_episode_idx:06d}.mp4"

    data_chunk_dir = out_root / "data" / chunk_name
    videos_chunk_dir = out_root / "videos" / chunk_name
    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    videos_chunk_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        qpos = f["observations/qpos"][...]
        action = f["action"][...]

        if qpos.ndim != 2 or action.ndim != 2:
            raise ValueError(f"Expected 2D qpos/action in {h5_path}")

        timesteps = int(qpos.shape[0])
        state_dim = int(qpos.shape[1])
        action_dim = int(action.shape[1])

        df = build_episode_dataframe(qpos, action, out_episode_idx, fps)
        df.to_parquet(data_chunk_dir / parquet_name, index=False)

        for cam in CAMERA_KEYS:
            h5_key = f"observations/images/{cam}"
            if h5_key not in f:
                raise KeyError(f"Missing camera key '{h5_key}' in {h5_path}")
            frames = f[h5_key][...]
            cam_dir = videos_chunk_dir / f"observation.images.{cam}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            write_video_mp4(frames, cam_dir / video_name, fps=fps)

    return timesteps, state_dim, action_dim


def build_info_json(
    total_episodes: int,
    total_frames: int,
    fps: float,
    state_dim: int,
    action_dim: int,
) -> Dict:
    return {
        "codebase_version": "v2.0",
        "robot_type": "xdof",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(CAMERA_KEYS),
        "chunks_size": CHUNK_SIZE,
        "fps": float(fps),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/"
            "{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.cam_high": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": "mpeg4",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.cam_left_wrist": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": "mpeg4",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.cam_right_wrist": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": "mpeg4",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
            },
            "annotation.task": {
                "dtype": "string",
                "shape": [1],
                "names": None,
            },
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def maybe_clear_old_outputs(output_dir: Path, force: bool) -> None:
    if not force:
        return
    # Only clear generated files under expected leaves; keep top-level dirs.
    for sub in ("data", "videos", "meta"):
        subdir = output_dir / sub
        if not subdir.exists():
            continue
        for child in subdir.iterdir():
            if child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink()
                for nested_dir in sorted([p for p in child.rglob("*") if p.is_dir()], reverse=True):
                    nested_dir.rmdir()
                child.rmdir()
            else:
                child.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ALOHA HDF5 episodes to minimal LeRobot v2 dataset format."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/data/user/wsong890/shuaizhou/dreamzero/cobot_data/cooking2"),
        help="Directory containing episode_*.hdf5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/user/wsong890/shuaizhou/dreamzero/cobot_data/cobot_dataset"),
        help="LeRobot-style output root (must contain or allow creating data/videos/meta)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS used for video writing and timestamps",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for quick testing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing generated outputs under data/videos/meta first",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_output_layout(output_dir)
    maybe_clear_old_outputs(output_dir, force=args.force)

    h5_paths = get_hdf5_paths(input_dir)
    if args.max_episodes is not None:
        h5_paths = h5_paths[: args.max_episodes]
    if not h5_paths:
        print("No episodes selected.", file=sys.stderr)
        sys.exit(1)

    total_frames = 0
    state_dim = None
    action_dim = None

    for out_ep_idx, h5_path in enumerate(h5_paths):
        print(f"[{out_ep_idx + 1}/{len(h5_paths)}] converting {h5_path.name}")
        t, sdim, adim = convert_episode(h5_path, output_dir, out_ep_idx, args.fps)
        total_frames += t
        if state_dim is None:
            state_dim = sdim
        if action_dim is None:
            action_dim = adim
        if state_dim != sdim or action_dim != adim:
            raise ValueError(
                f"Inconsistent dims at {h5_path.name}: "
                f"state {sdim} vs {state_dim}, action {adim} vs {action_dim}"
            )

    info = build_info_json(
        total_episodes=len(h5_paths),
        total_frames=total_frames,
        fps=args.fps,
        state_dim=int(state_dim),
        action_dim=int(action_dim),
    )
    info_path = output_dir / "meta" / "info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\nDone.")
    print(f"- input_dir : {input_dir}")
    print(f"- output_dir: {output_dir}")
    print(f"- episodes  : {len(h5_paths)}")
    print(f"- info.json : {info_path}")
    print("\nNext:")
    print(
        "python scripts/data/convert_lerobot_to_gear.py "
        f"--dataset-path \"{output_dir}\" "
        "--embodiment-tag xdof "
        f"--state-keys '{{\"joint_pos\":[0,{state_dim-1}],\"gripper_pos\":[{state_dim-1},{state_dim}]}}' "
        f"--action-keys '{{\"joint_pos\":[0,{action_dim-1}],\"gripper_pos\":[{action_dim-1},{action_dim}]}}' "
        "--relative-action-keys joint_pos gripper_pos "
        "--task-key annotation.task"
    )


if __name__ == "__main__":
    main()
