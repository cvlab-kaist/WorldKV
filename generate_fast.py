import argparse
import logging
import os
import re
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.modules.kv_modes import validate_kv_mode_flags
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import merge_video_audio, save_video, str2bool


EXAMPLE_PROMPT = {
    "i2v-A14B": {
        "prompt":
            "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder.",
        "image":
            "examples/04/image.jpg",
    },
}


def _resolve_control_npy_for_name(action_path, canonical_name, alt_suffix):
    """Resolve control npy path for naming, matching runtime loading behavior."""
    if action_path is None or not os.path.isdir(action_path):
        return None

    canonical_path = os.path.join(action_path, canonical_name)
    if os.path.exists(canonical_path):
        return canonical_path

    candidates = sorted(
        os.path.join(action_path, fname)
        for fname in os.listdir(action_path)
        if fname.endswith(alt_suffix)
    )
    if len(candidates) == 1:
        return candidates[0]
    return None


def _parse_traj_and_frames_from_action_path(action_path):
    """
    Parse trajectory name and frame count from:
      <trajectory>_<frames>_poses.npy or <trajectory>_<frames>_intrinsics.npy
    """
    poses_path = _resolve_control_npy_for_name(action_path, "poses.npy", "_poses.npy")
    intr_path = _resolve_control_npy_for_name(action_path, "intrinsics.npy", "_intrinsics.npy")

    stems = []
    if poses_path and poses_path.endswith("_poses.npy"):
        stems.append(os.path.basename(poses_path)[:-len("_poses.npy")])
    if intr_path and intr_path.endswith("_intrinsics.npy"):
        stems.append(os.path.basename(intr_path)[:-len("_intrinsics.npy")])

    for stem in stems:
        match = re.fullmatch(r"(.+)_([0-9]+)", stem)
        if match:
            return match.group(1), int(match.group(2))
    return None, None


def _parse_sample_from_image_path(image_path):
    """Parse sample id from image path (e.g. examples/03_RLRL120/image.jpg -> 03)."""
    if image_path is None:
        return None
    parent_name = os.path.basename(os.path.dirname(image_path))
    match = re.match(r"([0-9]+)", parent_name)
    if match:
        return match.group(1)

    # Fallback for flat sample layouts, e.g. examples/new/03.png
    file_stem = os.path.splitext(os.path.basename(image_path))[0]
    match = re.match(r"([0-9]+)", file_stem)
    return match.group(1) if match else None


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

    validate_kv_mode_flags(
        full_kv=args.full_kv,
        sliding_window=args.sliding_window,
        use_retrieval=args.use_retrieval,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="The camera path to generate the video from.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--max_attention_size",
        type=int,
        default=None,
        help="The size of kv cache during inference.")
    parser.add_argument(
        "--use_retrieval",
        action="store_true",
        default=False,
        help="Enable camera-based KV cache retrieval ([sink|retrieval|recent] window).")
    parser.add_argument(
        "--retrieval_frames",
        type=int,
        default=6,
        help="Number of latent frames to retrieve from KV bank (should be multiple of chunk_size).")
    parser.add_argument(
        "--retrieval_rope_correction",
        action="store_true",
        default=False,
        help="Enable RoPE correction for retrieved KV (rebase to contiguous positions before recent).")
    parser.add_argument(
        "--full_kv",
        action="store_true",
        default=False,
        help="Use full KV cache (no sliding window, no retrieval). Overrides --use_retrieval.")
    parser.add_argument(
        "--sliding_window",
        type=int,
        default=0,
        help="Pure FIFO sliding window of N latent frames (sink=0, no retrieval). 0 disables.")
    parser.add_argument(
        "--kv_compression_enable",
        action="store_true",
        default=False,
        help="Enable anchor+novelty chunk-wise token pruning on retrieval KV.")
    parser.add_argument(
        "--kv_compression_keep_ratio",
        type=float,
        default=0.5,
        help="Fraction of tokens kept per non-anchor frame when compression is enabled.")
    parser.add_argument(
        "--kv_compression_anchor_rotate",
        action="store_true",
        default=False,
        help="Rotate the anchor frame offset per retrieved chunk (offset = chunk_order %% chunk_size).")
    parser.add_argument(
        "--kv_compression_at_store",
        action="store_true",
        default=False,
        help="Compress each chunk's KV (anchor=frame 0, novelty pruning) at bank-store time "
             "to bound CPU RAM growth. Requires --kv_compression_enable. Forces anchor_rotate=False "
             "at store time; retrieval-time compression is then skipped.")
    parser.add_argument(
        "--kv_compression_pooled",
        action="store_true",
        default=False,
        help="Select kept tokens jointly across all non-anchor frames of a chunk "
             "(shared budget) instead of a fixed keep_per_frame from each frame. "
             "Currently wired for the --kv_compression_at_store path only.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default='output',
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--kv_bank_on_gpu",
        action="store_true",
        default=False,
        help="Keep retrieval KV bank on GPU (disable CPU offload). Requires ample VRAM.")
    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            input_prompt = args.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")
    
    logging.info("Creating WanI2VFast pipeline.")
    wan_i2v = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    logging.info("Generating video ...")
    video = wan_i2v.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        chunk_size=3,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        seed=args.base_seed,
        offload_model=args.offload_model,
        max_attention_size=args.max_attention_size,
        use_retrieval=args.use_retrieval,
        retrieval_frames=args.retrieval_frames,
        retrieval_rope_correction=args.retrieval_rope_correction,
        full_kv=args.full_kv,
        sliding_window=args.sliding_window,
        kv_compression_enable=args.kv_compression_enable,
        kv_compression_keep_ratio=args.kv_compression_keep_ratio,
        kv_compression_anchor_rotate=args.kv_compression_anchor_rotate,
        kv_compression_at_store=args.kv_compression_at_store,
        kv_compression_pooled=args.kv_compression_pooled,
        kv_bank_on_gpu=args.kv_bank_on_gpu)

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        if args.save_file is None:
            sample_id = _parse_sample_from_image_path(args.image)
            traj_name, frame_count = _parse_traj_and_frames_from_action_path(args.action_path)
            if sample_id is not None and traj_name is not None and frame_count is not None:
                args.save_file = os.path.join(
                    args.save_dir, f"{sample_id}_{traj_name}_{frame_count}.mp4")
            else:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                         "_")[:50]
                suffix = '.mp4'
                args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
                args.save_file = f'{args.save_dir}/{args.save_file}'

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
