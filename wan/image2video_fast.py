import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
# import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward_causal, sp_dit_forward_causal
from .distributed.util import get_world_size
from .modules.model_fast import WanModelFast, _compress_retrieval_kv
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE

from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from einops import rearrange

# trajectory file name compatibility
def _resolve_control_npy(action_path: str, canonical_name: str, alt_suffix: str) -> str:
    """
    Resolve a control npy file under action_path.
    Priority:
      1) canonical filename (e.g. poses.npy)
      2) exactly one file matching alt_suffix (e.g. *_poses.npy)
    """
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
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Cannot find '{canonical_name}' in '{action_path}', and no file matching '*{alt_suffix}'."
        )
    raise ValueError(
        f"Multiple files matching '*{alt_suffix}' in '{action_path}': {candidates}. "
        f"Keep a single match or provide canonical '{canonical_name}'."
    )


class WanI2VFast:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        pipe_dtype: torch.dtype = torch.bfloat16,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype
        self.pipe_dtype = pipe_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        if 'cam' in checkpoint_dir:
            self.control_type = 'cam'
        elif 'act' in checkpoint_dir:
            self.control_type = 'act'

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModelFast from {checkpoint_dir}")
        self.model = WanModelFast.from_pretrained(
            checkpoint_dir, subfolder=config.fast_noise_checkpoint, torch_dtype=torch.bfloat16, control_type=self.control_type)
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype).to(self.device)

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_causal, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward_causal, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, F, H, W]
        xt: the input noisy data with shape [B, C, F, H, W]
        timestep: the timestep with shape [B]
    
        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
    
        return x0_pred.to(original_dtype)


    def generate(self,
                 input_prompt,
                 img,
                 action_path=None,
                 chunk_size=3,
                 max_area=480 * 832,
                 frame_num=81,
                 timesteps_index=[0, 179, 358, 679],
                 shift=5.0,
                 seed=-1,
                 offload_model=True,
                 max_sequence_length=512,
                 max_attention_size=None,
                 use_retrieval=False,
                 retrieval_frames=6,
                 retrieval_rope_correction=False,
                 full_kv=False,
                 sliding_window=0,
                 kv_compression_enable=False,
                 kv_compression_keep_ratio=0.5,
                 kv_compression_anchor_rotate=False,
                 kv_compression_at_store=False,
                 kv_compression_pooled=False,
                 kv_bank_on_gpu=False,):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        if input_prompt is not None and isinstance(input_prompt, str):
            batch_size = 1
        elif input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1
        if action_path is not None:
            poses_path = _resolve_control_npy(action_path, "poses.npy", "_poses.npy")
            c2ws = np.load(poses_path) # opencv coordinate
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]
            if self.control_type == 'act':
                # In 'act' mode, use rotation of c2ws to control orientation and wasd_action to drive movement.
                action_path_npy = _resolve_control_npy(action_path, "action.npy", "_action.npy")
                wasd_action = np.load(action_path_npy) # wasd action
                wasd_action = wasd_action[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            lat_f,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
        timesteps = self.scheduler.timesteps[timesteps_index]

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]

        # cam preparation (only if action_path is provided)
        dit_cond_dict = None
        if action_path is not None:
            intrinsics_path = _resolve_control_npy(action_path, "intrinsics.npy", "_intrinsics.npy")
            Ks = torch.from_numpy(np.load(intrinsics_path)).float()

            # The provided intrinsics are for original image size (480p). We need to transform them according to the new image size (h, w).
            Ks = get_Ks_transformed(Ks,
                                    height_org=480,
                                    width_org=832,
                                    height_resize=h,
                                    width_resize=w,
                                    height_final=h,
                                    width_final=w)
            Ks = Ks[0]
            
            len_c2ws = len(c2ws)
            len_c2ws_ = int((len_c2ws - 1) // 4) + 1
            len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
            )
            c2ws_infer_abs = c2ws_infer.clone()  # save absolute poses for retrieval
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            if self.control_type == 'act':
                wasd_action = torch.from_numpy(wasd_action[::4]).float().to(self.device)
            else:
                wasd_action = None
            only_rays_d = wasd_action is not None
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w, only_rays_d=only_rays_d)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...] # [b, f*h*w, c]
            c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            if wasd_action is not None:
                wasd_action_tensor = wasd_action[:, None, None, :].repeat(1, h, w, 1) # [f, h, w, 3]
                wasd_action_tensor = rearrange(
                    wasd_action_tensor,
                    'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                    c1=int(h // lat_h),
                    c2=int(w // lat_w),
                )
                wasd_action_tensor = wasd_action_tensor[None, ...] # [b, f*h*w, c]
                wasd_action_tensor = rearrange(wasd_action_tensor, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
                c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)
        
        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

        # Initialize KV cache to all zeros
        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1]// 4)
        # Window = sink + retrieval + recent(6, fixed)
        if full_kv:
            use_retrieval = False
            window_size = lat_f
            kv_size = frame_seqlen * lat_f
            effective_local_attn_size = -1
            effective_sink_size = model_args.sink_size
        elif sliding_window > 0:
            use_retrieval = False
            window_size = sliding_window
            kv_size = frame_seqlen * min(window_size, lat_f)
            effective_local_attn_size = window_size
            effective_sink_size = 0  # pure FIFO
        else:
            retrieval_window = retrieval_frames if use_retrieval else 0
            recent_size = 6
            window_size = model_args.sink_size + retrieval_window + recent_size
            kv_size = frame_seqlen * min(window_size, lat_f)
            effective_local_attn_size = window_size
            effective_sink_size = model_args.sink_size

        # Update model's local_attn_size dynamically
        self.model.local_attn_size = effective_local_attn_size
        for block in self.model.blocks:
            block.local_attn_size = effective_local_attn_size
            block.self_attn.local_attn_size = effective_local_attn_size
            block.self_attn.sink_size = effective_sink_size
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
        self_kv_cache = self._initialize_self_kv_cache(num_layers=model_args.num_layers,
                                                      shape=self_kv_shape,
                                                      dtype=transformer_dtype,
                                                      device=self.device)
        # Separate bank storage (outside kv_cache dicts to avoid FSDP reference issues)
        kv_bank_k = [[] for _ in range(model_args.num_layers)]  # per-layer list of chunk KVs
        kv_bank_v = [[] for _ in range(model_args.num_layers)]
        cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]
        cross_kv_cache = self._initialize_crossattn_cache(num_layers=model_args.num_layers,
                                                         shape=cross_kv_shape,
                                                         dtype=transformer_dtype,
                                                         device=self.device)
        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            # sample videos
            latent = noise
            latents_chunk = latent.split(chunk_size, dim=1) # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []
            for chunk_id in tqdm(range(num_inference_chunk)):
                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]

                dit_cond_dict = {
                    "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                }
    
                # Retrieval: select chunks from bank by camera pose similarity.
                retrieval_kv = None
                if use_retrieval and action_path is not None and len(kv_bank_k[0]) > 0:
                    model_sink_size = getattr(self.model.config, 'sink_size', 0)
                    sink_chunks = (model_sink_size + chunk_size - 1) // chunk_size
                    recent_frames = max(0, kv_size // frame_seqlen - model_sink_size - retrieval_frames)
                    recent_chunks = max(1, recent_frames // chunk_size)
                    num_retrieve_chunks = retrieval_frames // chunk_size

                    # Current chunk average pose (absolute)
                    cs = chunk_id * chunk_size
                    ce = min(cs + chunk_size, c2ws_infer_abs.shape[0])
                    current_trans = c2ws_infer_abs[cs:ce, :3, 3].mean(dim=0)
                    current_rot = c2ws_infer_abs[cs:ce, :3, :3].mean(dim=0)

                    # Candidate chunks: exclude sink and recent
                    recent_start_chunk = max(sink_chunks, chunk_id - recent_chunks)
                    candidate_ids = []
                    candidate_trans = []
                    candidate_rots = []
                    for cid in range(sink_chunks, recent_start_chunk):
                        if cid < len(kv_bank_k[0]):
                            s = cid * chunk_size
                            e = min(s + chunk_size, c2ws_infer_abs.shape[0])
                            candidate_ids.append(cid)
                            candidate_trans.append(c2ws_infer_abs[s:e, :3, 3].mean(dim=0))
                            candidate_rots.append(c2ws_infer_abs[s:e, :3, :3].mean(dim=0))

                    if candidate_ids and num_retrieve_chunks > 0:
                        # Translation L2 distance
                        candidate_trans = torch.stack(candidate_trans).to(current_trans.device)
                        trans_dist = ((candidate_trans - current_trans) ** 2).sum(dim=-1)

                        # Rotation geodesic distance: arccos((trace(R1^T @ R2) - 1) / 2)
                        candidate_rots = torch.stack(candidate_rots).to(current_rot.device)
                        rel_rot = candidate_rots.transpose(-1, -2) @ current_rot.unsqueeze(0)
                        trace_val = rel_rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [N]
                        cos_angle = (trace_val - 1.0) / 2.0
                        rot_dist = torch.acos(cos_angle.clamp(-1.0, 1.0))  # [N], radians

                        # Normalize each to [0, 1] then combine
                        trans_norm = trans_dist / trans_dist.max() if trans_dist.max() > 0 else trans_dist
                        rot_norm = rot_dist / rot_dist.max() if rot_dist.max() > 0 else rot_dist
                        distances = 0.5 * trans_norm + 0.5 * rot_norm

                        k = min(num_retrieve_chunks, len(candidate_ids))
                        _, top_idx = distances.topk(k, largest=False)
                        selected_ids = [candidate_ids[i] for i in top_idx.tolist()]
                        retr_frames = [sid * chunk_size for sid in selected_ids]
                        # Compute window frame indices
                        sink_f = list(range(model_sink_size))
                        retr_f = []
                        for sid in selected_ids:
                            retr_f.extend(range(sid * chunk_size, sid * chunk_size + chunk_size))
                        recent_start = max(model_sink_size, chunk_id * chunk_size - recent_frames)
                        recent_f = list(range(recent_start, ce))
                        # print(
                        #     f"[Chunk {chunk_id}/{num_inference_chunk}] "
                        #     f"Generating frames {cs}-{ce-1} | "
                        #     f"Window: sink={sink_f} retr={retr_f} recent={recent_f} | "
                        #     f"trans_dist={[f'{d:.4f}' for d in trans_dist[top_idx].tolist()]} "
                        #     f"rot_dist_deg={[f'{d:.1f}' for d in torch.rad2deg(rot_dist[top_idx]).tolist()]}",
                        #     flush=True)

                        # Source frame IDs for RoPE correction
                        src_frame_ids = [sid * chunk_size for sid in selected_ids]

                        retrieval_kv = []
                        for layer_idx in range(model_args.num_layers):
                            retr_k = torch.cat([kv_bank_k[layer_idx][i].to(self.device) for i in selected_ids], dim=1)
                            retr_v = torch.cat([kv_bank_v[layer_idx][i].to(self.device) for i in selected_ids], dim=1)
                            # If compression was applied at store time, skip retrieval-time
                            # compression; the bank entries are already pruned.
                            runtime_compress = (kv_compression_enable and not kv_compression_at_store)
                            retrieval_kv.append({
                                'k': retr_k, 'v': retr_v,
                                'src_frame_ids': src_frame_ids,
                                'rope_correction': retrieval_rope_correction,
                                'compress': runtime_compress,
                                'compress_keep_ratio': kv_compression_keep_ratio,
                                'compress_anchor_rotate': kv_compression_anchor_rotate,
                                'compress_chunk_size': chunk_size,
                                'stored_compressed': (kv_compression_enable and kv_compression_at_store),
                                'recent_frames': recent_frames,
                            })

                if use_retrieval and retrieval_kv is None:
                    cs_dbg = chunk_id * chunk_size
                    ce_dbg = min(cs_dbg + chunk_size, lat_f)
                    bank_len = len(kv_bank_k[0])
                    model_sink_size_dbg = getattr(self.model.config, 'sink_size', 0)
                    window_end = ce_dbg
                    window_start = max(0, cs_dbg - (kv_size // frame_seqlen - chunk_size))
                    window_f = list(range(window_start, window_end))
                    # print(
                    #     f"[Chunk {chunk_id}/{num_inference_chunk}] "
                    #     f"Generating frames {cs_dbg}-{ce_dbg-1} | "
                    #     f"No retrieval (bank={bank_len}) | "
                    #     f"Window: {window_f}",
                    #     flush=True)

                effective_max_attn = kv_size if max_attention_size is None else max_attention_size
                kwargs = {
                    'context': [context[0]],
                    'seq_len': max_seq_len,
                    'y': [current_condition],
                    'dit_cond_dict': dit_cond_dict,
                    'kv_cache': self_kv_cache,
                    'crossattn_cache': cross_kv_cache,
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': effective_max_attn,
                    'retrieval_kv': retrieval_kv,
                }

                if offload_model:
                    torch.cuda.empty_cache()

                for timestep_idx in range(len(timesteps)):
                    latent_model_input = [current_latent.to(self.device)]
                    current_timestep = [timesteps[timestep_idx]]

                    timestep = torch.stack(current_timestep).to(self.device)

                    noise_pred = self.model(
                        x=latent_model_input, t=timestep, **kwargs)[0]

                    if offload_model:
                        torch.cuda.empty_cache()

                    x0 = self._convert_flow_pred_to_x0(
                        flow_pred=noise_pred,
                        xt=current_latent,
                        timestep=current_timestep[0],
                        scheduler=self.scheduler,
                    )

                    if timestep_idx < len(timesteps) - 1:
                        next_timestep = timesteps[timestep_idx + 1]
                        current_latent = self.scheduler.add_noise(x0, torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype), next_timestep)
                    else:
                        # note return x0
                        break

                pred_latent_chunks.append(x0)

                # Update kv cache with clean t=0 pass
                context_timestep = [timesteps[-1] * 0.0]
                timestep = torch.stack(context_timestep).to(self.device)
                self.model(x=[x0], t=timestep, **kwargs)

                # Store current chunk's clean KV to bank (extract directly from kv_cache)
                if use_retrieval:
                    chunk_tokens = chunk_size * frame_seqlen
                    bank_device = self.device if kv_bank_on_gpu else torch.device('cpu')
                    store_compress = (kv_compression_enable and kv_compression_at_store)
                    for layer_idx, lc in enumerate(self_kv_cache):
                        local_end = lc['local_end_index'].item()
                        k_slice = lc['k'][:, local_end - chunk_tokens:local_end].detach()
                        v_slice = lc['v'][:, local_end - chunk_tokens:local_end].detach()
                        if store_compress:
                            # Anchor-at-offset-0 + novelty pruning applied once at store time.
                            # sp_reduce ensures every rank keeps the same token indices under
                            # head-split SP so the bank entries stay length-consistent.
                            k_slice, v_slice = _compress_retrieval_kv(
                                k_slice, v_slice,
                                chunk_size=chunk_size,
                                frame_seqlen=frame_seqlen,
                                keep_ratio=kv_compression_keep_ratio,
                                anchor_rotate=False,
                                sp_reduce=(self.sp_size > 1),
                                pooled=kv_compression_pooled,
                            )
                        kv_bank_k[layer_idx].append(k_slice.to(bank_device).clone())
                        kv_bank_v[layer_idx].append(v_slice.to(bank_device).clone())
                    if store_compress and chunk_id == 0:
                        stored_tokens = kv_bank_k[0][0].shape[1]
                        _mode = "pooled" if kv_compression_pooled else "per-frame"
                    #     print(f"[Debug] store-time compression on ({_mode}): {chunk_tokens} -> {stored_tokens} tokens/chunk/layer", flush=True)
                    # print(f"[Debug] After clean pass chunk {chunk_id}: bank len={len(kv_bank_k[0])} device={bank_device}", flush=True)

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode([pred_latent_chunks])

        # del noise, latent, x0
        # del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def _initialize_self_kv_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a Per-GPU KV cache for the SelfAttn.
        """
        self_kv_cache = []
        for _ in range(num_layers):
            self_kv_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device)
            })

        return self_kv_cache


    def _initialize_crossattn_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the CrossAttb
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': False
            })

        return crossattn_cache
