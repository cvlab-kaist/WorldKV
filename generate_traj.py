#!/usr/bin/env python3
import argparse
import os
import re
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

"""
python generate_traj.py --frames 201 --trajectory R_L_R_L90 --noop-frames 16

RL180 : 0th, 179th frame
WLR120 : 32th, 151th frame
WLWR180 : 0th, 211th frame
"""

DEFAULT_INTRINSIC = (415.5298, 415.6922, 415.77786, 239.77779)
ROTATION_DEG_PER_FRAME = 2.0
TRANSLATION_FRAMES_PER_OP = 24


def parse_sequence(sequence: str) -> Tuple[List[str], float]:
    seq = sequence.strip().replace(" ", "")
    # trajectory format:
    # - Operations: R, L, W, S, A, D, _
    # - Optional trailing degree number used by R/L ops only.
    #   examples:
    #     R_L_R_L90, W_A_D_S, R_W_L120
    m = re.fullmatch(r"([RLSWAD_]+)(\d+(?:\.\d+)?)?", seq)
    if not m:
        raise ValueError(
            "Invalid trajectory format. Example: LRLR45, R_L_R_L90, W_A_D_S, R_W_L120 "
            "(operations R/L/W/S/A/D/_ with optional trailing degree number)."
        )
    ops = list(m.group(1))
    degree_group = m.group(2)
    degree = float(degree_group) if degree_group is not None else 0.0
    if not ops:
        raise ValueError("Trajectory operation list is empty.")
    if degree < 0:
        raise ValueError("Rotation degree must be non-negative.")
    if any(op in ("R", "L") for op in ops) and degree_group is None:
        raise ValueError(
            "Trajectory includes R/L operations, so a trailing degree is required "
            "(example: R_L_W90)."
        )
    return ops, degree


def parse_noop_frames(noop_frames: str, num_noop: int) -> List[int]:
    if num_noop == 0:
        return []
    if noop_frames is None:
        raise ValueError(
            "Trajectory contains '_' operations, so --noop-frames is required."
        )

    raw = noop_frames.strip()
    if "," in raw:
        vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if len(vals) != num_noop:
            raise ValueError(
                f"--noop-frames expects {num_noop} values for this sequence, got {len(vals)}."
            )
    else:
        vals = [int(raw)] * num_noop

    if any(v <= 0 for v in vals):
        raise ValueError("All no-op frame counts must be positive integers.")
    return vals


def build_yaw_series(
    ops: Sequence[str],
    degree: float,
    noop_frame_counts: Sequence[int],
    translation_distance: float,
    rotation_step_deg: float = ROTATION_DEG_PER_FRAME,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], List[int]]:
    if rotation_step_deg <= 0:
        raise ValueError("rotation_step_deg must be positive.")
    if any(op in ("R", "L") for op in ops) and degree == 0:
        raise ValueError("Rotation degree must be > 0.")
    if translation_distance < 0:
        raise ValueError("translation_distance must be non-negative.")

    has_rotation = any(op in ("R", "L") for op in ops)
    if has_rotation:
        # Enforce fixed 2 deg/frame and exact same rotation amount for every R/L op.
        steps_per_rot_float = degree / rotation_step_deg
        steps_per_rot = int(round(steps_per_rot_float))
        if not np.isclose(steps_per_rot_float, steps_per_rot):
            raise ValueError(
                f"Degree {degree} is not divisible by fixed rotation step "
                f"{rotation_step_deg} deg/frame."
            )
    else:
        steps_per_rot = 0

    yaw_values: List[float] = []
    xyz_values: List[List[float]] = []
    op_deltas: List[float] = []
    op_trans_deltas: List[float] = []
    op_frames: List[int] = []
    current = 0.0
    current_xyz = np.zeros((3,), dtype=np.float32)
    noop_i = 0

    for op in ops:
        if op in ("R", "L"):
            sign = 1.0 if op == "R" else -1.0
            for _ in range(steps_per_rot):
                current += sign * rotation_step_deg
                yaw_values.append(current)
                xyz_values.append(current_xyz.tolist())
            op_deltas.append(sign * rotation_step_deg * steps_per_rot)
            op_trans_deltas.append(0.0)
            op_frames.append(steps_per_rot)
        elif op in ("W", "S", "A", "D"):
            trans_sign = 1.0 if op in ("W", "D") else -1.0
            trans_step = translation_distance / TRANSLATION_FRAMES_PER_OP
            for _ in range(TRANSLATION_FRAMES_PER_OP):
                yaw_rad = np.deg2rad(current)
                forward = np.array(
                    [np.sin(yaw_rad), 0.0, np.cos(yaw_rad)],
                    dtype=np.float32,
                )
                right = np.array(
                    [np.cos(yaw_rad), 0.0, -np.sin(yaw_rad)],
                    dtype=np.float32,
                )
                direction = forward if op in ("W", "S") else right
                current_xyz = current_xyz + (trans_sign * trans_step) * direction
                yaw_values.append(current)
                xyz_values.append(current_xyz.tolist())
            op_deltas.append(0.0)
            op_trans_deltas.append(trans_sign * translation_distance)
            op_frames.append(TRANSLATION_FRAMES_PER_OP)
        else:
            n = noop_frame_counts[noop_i]
            noop_i += 1
            for _ in range(n):
                yaw_values.append(current)
                xyz_values.append(current_xyz.tolist())
            op_deltas.append(0.0)
            op_trans_deltas.append(0.0)
            op_frames.append(n)

    return (
        np.array(yaw_values, dtype=np.float32),
        np.array(xyz_values, dtype=np.float32),
        op_deltas,
        op_trans_deltas,
        op_frames,
    )


def make_poses(yaw_deg: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    n = yaw_deg.shape[0]
    # mats = R.from_euler("y", yaw_deg, degrees=True).as_matrix().astype(np.float32)
    mats = R.from_euler("y", yaw_deg[:, None], degrees=True).as_matrix().astype(np.float32)
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    poses[:, :3, :3] = mats
    poses[:, :3, 3] = xyz
    poses[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return poses


def make_intrinsics(num_frames: int, intrinsic: Tuple[float, float, float, float]) -> np.ndarray:
    return np.tile(np.array(intrinsic, dtype=np.float32), (num_frames, 1))


def sanitize_name(sequence: str) -> str:
    return sequence.replace(" ", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LingBot-World compatible trajectory npy files."
    )
    parser.add_argument(
        "--frames",
        type=int,
        required=True,
        help="Total number of frames to generate.",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        required=True,
        help="Trajectory sequence like LRLR45 or R_L_R_L90.",
    )
    parser.add_argument(
        "--noop-frames",
        type=str,
        default=None,
        help=(
            "Required when '_' exists in trajectory. "
            "Either a single int applied to all '_' ops, or comma-separated ints per '_' in order."
        ),
    )
    parser.add_argument("--fx", type=float, default=DEFAULT_INTRINSIC[0])
    parser.add_argument("--fy", type=float, default=DEFAULT_INTRINSIC[1])
    parser.add_argument("--cx", type=float, default=DEFAULT_INTRINSIC[2])
    parser.add_argument("--cy", type=float, default=DEFAULT_INTRINSIC[3])
    parser.add_argument(
        "--translation-distance",
        type=float,
        default=1.0,
        help=(
            "Total displacement per W/S/A/D op. "
            f"Each translational op always spans {TRANSLATION_FRAMES_PER_OP} frames."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="traj",
        help="Directory to save output npy files.",
    )
    args = parser.parse_args()

    ops, degree = parse_sequence(args.trajectory)
    if args.frames <= 0:
        raise ValueError("--frames must be a positive integer.")

    noop_count = sum(1 for op in ops if op == "_")
    noop_frame_counts = parse_noop_frames(args.noop_frames, noop_count)

    yaw, xyz, op_deltas, op_trans_deltas, op_frames = build_yaw_series(
        ops=ops,
        degree=degree,
        noop_frame_counts=noop_frame_counts,
        translation_distance=args.translation_distance,
        rotation_step_deg=ROTATION_DEG_PER_FRAME,
    )

    if yaw.shape[0] > args.frames:
        raise ValueError(
            f"Trajectory consumes {yaw.shape[0]} frames, but --frames={args.frames}. "
            "Increase --frames or reduce trajectory/no-op durations."
        )

    # Fill remaining frames with no-operation at the end.
    remaining_noop = args.frames - yaw.shape[0]
    if remaining_noop > 0:
        hold = yaw[-1] if yaw.size > 0 else 0.0
        yaw = np.concatenate(
            [yaw, np.full((remaining_noop,), hold, dtype=np.float32)],
            axis=0,
        )
        hold_xyz = xyz[-1] if xyz.size > 0 else np.zeros((3,), dtype=np.float32)
        xyz = np.concatenate(
            [xyz, np.tile(hold_xyz[None, :], (remaining_noop, 1)).astype(np.float32)],
            axis=0,
        )

    poses = make_poses(yaw, xyz)
    intrinsic = make_intrinsics(args.frames, (args.fx, args.fy, args.cx, args.cy))

    seq_name = sanitize_name(args.trajectory)
    run_name = f"{seq_name}_{args.frames}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    poses_path = os.path.join(run_dir, f"{run_name}_poses.npy")
    intr_path = os.path.join(run_dir, f"{run_name}_intrinsics.npy")

    np.save(poses_path, poses)
    np.save(intr_path, intrinsic)

    print(f"Saved: {poses_path} (shape={poses.shape}, dtype={poses.dtype})")
    print(f"Saved: {intr_path} (shape={intrinsic.shape}, dtype={intrinsic.dtype})")
    rot_abs = [abs(d) for d, op in zip(op_deltas, ops) if op in ("R", "L")]
    rotations_equal = all(np.isclose(v, degree) for v in rot_abs)
    print(
        f"Ops={ops}, degree_per_rotation={degree}, "
        f"rotation_step_deg_per_frame={ROTATION_DEG_PER_FRAME}, "
        f"translation_frames_per_op={TRANSLATION_FRAMES_PER_OP}, per_op_frames={op_frames}"
    )
    print(f"Rotation deltas by op (deg): {op_deltas}")
    print(f"Translation deltas by op (units): {op_trans_deltas}")
    print(f"Rotation magnitudes all equal to target degree: {rotations_equal}")
    print(f"Remaining end no-op frames appended: {remaining_noop}")
    print(f"Final yaw={float(yaw[-1]) if yaw.size else 0.0:.4f} deg")
    print(
        "Final xyz=["
        f"{float(xyz[-1, 0]) if xyz.size else 0.0:.4f}, "
        f"{float(xyz[-1, 1]) if xyz.size else 0.0:.4f}, "
        f"{float(xyz[-1, 2]) if xyz.size else 0.0:.4f}]"
    )


if __name__ == "__main__":
    main()
