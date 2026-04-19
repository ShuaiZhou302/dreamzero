#!/usr/bin/env python3
"""
Convert ALOHA-style HDF5 episodes to a minimal LeRobot v2 dataset layout.

Input (example):
  /path/to/cooking2/episode_0.hdf5
  /path/to/cooking2/episode_1.hdf5
  Or nested tasks, e.g. new_data/stack_somethings/episode_0.hdf5 (auto-discovered).

Per-episode language: if the HDF5 root has attribute ``task_description`` (or a few
other common keys), that string is written to ``annotation.task`` for every frame of
that episode. Otherwise falls back to the parent folder name (underscores → spaces)
or the input directory basename.

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

# Root-group HDF5 attributes tried in order for annotation.task (first non-empty wins).
TASK_ATTR_KEYS = (
    "task_description",
    "language_instruction",
    "instruction",
    "task",
    "task_name",
)


def _h5_attr_to_str(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s if s else None
    if isinstance(val, (bytes, bytearray)):
        s = bytes(val).decode("utf-8", errors="replace").strip()
        return s if s else None
    try:
        if hasattr(val, "item"):
            return _h5_attr_to_str(val.item())
    except Exception:
        pass
    try:
        arr = np.asarray(val)
        if arr.shape == ():
            return _h5_attr_to_str(arr.item())
    except Exception:
        pass
    s = str(val).strip()
    return s if s else None


def read_episode_instruction(h5f: h5py.File, fallback: str) -> str:
    for key in TASK_ATTR_KEYS:
        if key not in h5f.attrs:
            continue
        text = _h5_attr_to_str(h5f.attrs[key])
        if text:
            return text
    return fallback


def parse_episode_idx(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.hdf5$", path.name)
    if not match:
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(match.group(1))


def get_hdf5_paths(input_dir: Path, flat_only: bool) -> List[Path]:
    flat = sorted(input_dir.glob("episode_*.hdf5"), key=parse_episode_idx)
    if flat:
        return flat
    if flat_only:
        raise FileNotFoundError(
            f"No episode_*.hdf5 in {input_dir} (--flat-only: subdirectories not searched)."
        )
    deep = list(input_dir.rglob("episode_*.hdf5"))
    if not deep:
        raise FileNotFoundError(f"No episode_*.hdf5 under {input_dir}")
    return sorted(deep, key=lambda p: (str(p.parent), parse_episode_idx(p)))


def fallback_task_for_path(h5_path: Path, input_dir: Path) -> str:
    """When HDF5 has no task attrs: use task folder name, else dataset root name."""
    if h5_path.parent.resolve() != input_dir.resolve():
        s = h5_path.parent.name.replace("_", " ").strip()
        if s:
            return s
    return infer_task_text(input_dir, None)


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
    task_text: str,
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
            # Usually one instruction text for the whole episode.
            "annotation.task": [task_text] * t,
        }
    )
    return df


def convert_episode(
    h5_path: Path,
    out_root: Path,
    out_episode_idx: int,
    fps: float,
    task_text: str,
    use_h5_task_attrs: bool,
    dataset_input_dir: Path,
) -> Tuple[int, int, int, str]:
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

        if use_h5_task_attrs:
            fb = fallback_task_for_path(h5_path, dataset_input_dir)
            episode_task = read_episode_instruction(f, fb)
        else:
            episode_task = task_text

        df = build_episode_dataframe(
            qpos, action, out_episode_idx, fps, episode_task
        )
        df.to_parquet(data_chunk_dir / parquet_name, index=False)

        for cam in CAMERA_KEYS:
            h5_key = f"observations/images/{cam}"
            if h5_key not in f:
                raise KeyError(f"Missing camera key '{h5_key}' in {h5_path}")
            frames = f[h5_key][...]
            cam_dir = videos_chunk_dir / f"observation.images.{cam}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            write_video_mp4(frames, cam_dir / video_name, fps=fps)

    return timesteps, state_dim, action_dim, episode_task


def build_info_json(
    total_episodes: int,
    total_frames: int,
    fps: float,
    state_dim: int,
    action_dim: int,
    total_tasks: int = 1,
) -> Dict:
    return {
        "codebase_version": "v2.0",
        "robot_type": "xdof",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": int(total_tasks),
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


def infer_task_text(input_dir: Path, explicit_task_text: str | None) -> str:
    if explicit_task_text is not None and explicit_task_text.strip():
        return explicit_task_text.strip()
    # Default task text: dataset folder name, converting underscores to spaces.
    return input_dir.name.replace("_", " ").strip() or "task"


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
        "--task-text",
        type=str,
        default=None,
        help=(
            "If set, this exact string is used as annotation.task for every episode "
            "(ignores HDF5 task attributes)."
        ),
    )
    parser.add_argument(
        "--ignore-h5-task-attrs",
        action="store_true",
        help=(
            "Do not read task text from HDF5 root attributes; use the same fallback "
            "string for all episodes (input-dir basename, underscores → spaces, "
            "unless --task-text is set)."
        ),
    )
    parser.add_argument(
        "--flat-only",
        action="store_true",
        help="Only look for episode_*.hdf5 directly under --input-dir (no subdirectory search).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing generated outputs under data/videos/meta first",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    task_text = infer_task_text(input_dir, args.task_text)
    use_h5_task_attrs = args.task_text is None and not args.ignore_h5_task_attrs
    ensure_output_layout(output_dir)
    maybe_clear_old_outputs(output_dir, force=args.force)

    h5_paths = get_hdf5_paths(input_dir, flat_only=args.flat_only)
    if args.max_episodes is not None:
        h5_paths = h5_paths[: args.max_episodes]
    if not h5_paths:
        print("No episodes selected.", file=sys.stderr)
        sys.exit(1)

    total_frames = 0
    state_dim = None
    action_dim = None
    distinct_tasks: set[str] = set()

    for out_ep_idx, h5_path in enumerate(h5_paths):
        try:
            rel = h5_path.relative_to(input_dir)
        except ValueError:
            rel = h5_path
        print(f"[{out_ep_idx + 1}/{len(h5_paths)}] converting {rel}")
        t, sdim, adim, ep_task = convert_episode(
            h5_path,
            output_dir,
            out_ep_idx,
            args.fps,
            task_text,
            use_h5_task_attrs=use_h5_task_attrs,
            dataset_input_dir=input_dir,
        )
        distinct_tasks.add(ep_task)
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
        total_tasks=max(1, len(distinct_tasks)),
    )
    info_path = output_dir / "meta" / "info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\nDone.")
    print(f"- input_dir : {input_dir}")
    print(f"- output_dir: {output_dir}")
    print(f"- episodes  : {len(h5_paths)}")
    if use_h5_task_attrs:
        print(
            f"- annotation.task: per-episode from HDF5 root attrs when present; "
            f"{len(distinct_tasks)} distinct instruction(s)"
        )
    else:
        print(f"- annotation.task (same for all episodes): {task_text}")
    print(f"- info.json : {info_path}")
    print("\nNext:")
    print(
        "python scripts/data/convert_lerobot_to_gear.py "
        f"--dataset-path \"{output_dir}\" "
        "--embodiment-tag agx_aloha "
        "--state-keys '{\"left_joint_pos\":[0,6],\"left_gripper_pos\":[6,7],"
        "\"right_joint_pos\":[7,13],\"right_gripper_pos\":[13,14]}' "
        "--action-keys '{\"left_joint_pos\":[0,6],\"left_gripper_pos\":[6,7],"
        "\"right_joint_pos\":[7,13],\"right_gripper_pos\":[13,14]}' "
        "--relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos "
        "--task-key annotation.task"
    )


if __name__ == "__main__":
    main()
