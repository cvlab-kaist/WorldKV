"""Some of the functions are borrowed from SelfForcing (https://github.com/guandeh17/Self-Forcing)."""
import math
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    WanLayerNorm,
    WanSelfAttention,
    rope_params,
    sinusoidal_embedding_1d
)

from .attention import flash_attention


def _rope_time_delta_mul_(k_chunk, freqs, delta_frames):
    """In-place time-axis RoPE shift for already-roped keys.

    k_chunk: [B, L, H, D] where D is head_dim (real-valued RoPE-packed).
    freqs  : complex RoPE freqs, shape [max_seq_len, D/2].
    delta_frames: number of frames to shift (positive: forward, negative: backward).

    Only the *time* channel portion is rotated by exp(i * w * delta_frames).
    """
    if delta_frames == 0:
        return

    b, l, h, d = k_chunk.shape
    assert d % 2 == 0
    c = d // 2
    t_c = c - 2 * (c // 3)
    h_c = c // 3
    w_c = c // 3

    freqs_t, _, _ = freqs.split([t_c, h_c, w_c], dim=1)

    shift = abs(int(delta_frames))
    max_pos = freqs_t.shape[0] - 1
    if shift > max_pos:
        shift = max_pos

    mult = freqs_t[shift] if delta_frames >= 0 else torch.conj(freqs_t[shift])
    mult = mult.view(1, 1, 1, t_c)

    time_ri = k_chunk[..., : 2 * t_c]
    time_cx = torch.view_as_complex(time_ri.to(torch.float64).reshape(-1, t_c, 2))
    time_cx = time_cx * mult.to(time_cx.dtype)
    time_ri_new = torch.view_as_real(time_cx).reshape(b, l, h, t_c, 2).flatten(-2)
    time_ri.copy_(time_ri_new.to(time_ri.dtype))


def _compress_retrieval_kv(retr_k, retr_v, chunk_size, frame_seqlen,
                           keep_ratio, anchor_rotate, sp_reduce=False,
                           pooled=False):
    """Anchor + novelty chunk-wise token pruning for retrieval KV.

    retr_k, retr_v : [B, retr_tokens, H, D]  (H is local heads under SP).
    chunk_size     : frames per retrieved chunk.
    frame_seqlen   : tokens per frame.
    keep_ratio     : fraction of tokens kept per non-anchor frame.
    anchor_rotate  : if True, anchor_offset = chunk_order % chunk_size.
    sp_reduce      : if True, all-reduce similarity stats over the world group
                     so every rank picks the same tokens under head-split SP.
    pooled         : if True, select keep_total = (chunk_size-1)*keep_per_frame
                     tokens jointly across ALL non-anchor frames (shared budget)
                     instead of keep_per_frame from each frame independently. The
                     per-chunk total kept is identical either way, so downstream
                     token-count / RoPE-correction invariants are preserved.
    """
    B, total_tokens, H, D = retr_k.shape
    chunk_tokens = chunk_size * frame_seqlen
    if total_tokens == 0 or total_tokens % chunk_tokens != 0:
        return retr_k, retr_v
    keep_per_frame = max(1, int(math.ceil(keep_ratio * frame_seqlen)))
    if keep_per_frame >= frame_seqlen:
        return retr_k, retr_v

    num_chunks = total_tokens // chunk_tokens
    eps = 1e-8
    out_k_chunks = []
    out_v_chunks = []

    for ci in range(num_chunks):
        c_s = ci * chunk_tokens
        ck = retr_k[:, c_s:c_s + chunk_tokens].view(B, chunk_size, frame_seqlen, H, D)
        cv = retr_v[:, c_s:c_s + chunk_tokens].view(B, chunk_size, frame_seqlen, H, D)

        anchor_off = (ci % chunk_size) if anchor_rotate else 0
        anchor_k = ck[:, anchor_off]                 # [B, F_s, H, D]
        anchor_v = cv[:, anchor_off]
        centroid = anchor_k.float().mean(dim=1)      # [B, H, D]
        cen_sq = (centroid ** 2).sum(dim=(-2, -1))   # [B]
        if sp_reduce:
            import torch.distributed as dist
            dist.all_reduce(cen_sq, op=dist.ReduceOp.SUM)

        if pooled:
            # Pooled variant: pick keep_total tokens across ALL non-anchor frames
            # jointly (shared budget) rather than keep_per_frame from each frame.
            # keep_total matches the per-frame total so the per-chunk token count
            # is unchanged. RoPE correction applies a uniform per-chunk delta and
            # attention is permutation-invariant, so the [anchor | pooled] layout
            # is safe regardless of which frames the kept tokens came from.
            keep_total = (chunk_size - 1) * keep_per_frame
            if keep_total > 0:
                na_k = torch.cat(
                    [ck[:, fi] for fi in range(chunk_size) if fi != anchor_off], dim=1)
                na_v = torch.cat(
                    [cv[:, fi] for fi in range(chunk_size) if fi != anchor_off], dim=1)
                nk_f = na_k.float()
                dot = (nk_f * centroid.unsqueeze(1)).sum(dim=(-2, -1))  # [B, (cs-1)*F_s]
                tok_sq = (nk_f ** 2).sum(dim=(-2, -1))
                if sp_reduce:
                    import torch.distributed as dist
                    dist.all_reduce(dot, op=dist.ReduceOp.SUM)
                    dist.all_reduce(tok_sq, op=dist.ReduceOp.SUM)
                sim = dot / (torch.sqrt(tok_sq) * torch.sqrt(cen_sq).unsqueeze(-1) + eps)
                _, idx = sim.topk(keep_total, dim=-1, largest=False)
                idx, _ = idx.sort(dim=-1)
                gather_idx = idx[:, :, None, None].expand(-1, -1, H, D)
                kept_k = torch.gather(na_k, 1, gather_idx)
                kept_v = torch.gather(na_v, 1, gather_idx)
                out_k_chunks.append(torch.cat([anchor_k, kept_k], dim=1))
                out_v_chunks.append(torch.cat([anchor_v, kept_v], dim=1))
            else:
                out_k_chunks.append(anchor_k)
                out_v_chunks.append(anchor_v)
            continue

        frames_out_k = [None] * chunk_size
        frames_out_v = [None] * chunk_size
        frames_out_k[anchor_off] = anchor_k
        frames_out_v[anchor_off] = anchor_v

        for fi in range(chunk_size):
            if fi == anchor_off:
                continue
            frame_k = ck[:, fi]                      # [B, F_s, H, D]
            frame_v = cv[:, fi]
            fk_f = frame_k.float()
            dot = (fk_f * centroid.unsqueeze(1)).sum(dim=(-2, -1))  # [B, F_s]
            tok_sq = (fk_f ** 2).sum(dim=(-2, -1))                  # [B, F_s]
            if sp_reduce:
                import torch.distributed as dist
                dist.all_reduce(dot, op=dist.ReduceOp.SUM)
                dist.all_reduce(tok_sq, op=dist.ReduceOp.SUM)
            sim = dot / (torch.sqrt(tok_sq) * torch.sqrt(cen_sq).unsqueeze(-1) + eps)
            _, idx = sim.topk(keep_per_frame, dim=-1, largest=False)
            idx, _ = idx.sort(dim=-1)
            gather_idx = idx[:, :, None, None].expand(-1, -1, H, D)
            frames_out_k[fi] = torch.gather(frame_k, 1, gather_idx)
            frames_out_v[fi] = torch.gather(frame_v, 1, gather_idx)

        out_k_chunks.append(torch.cat(frames_out_k, dim=1))
        out_v_chunks.append(torch.cat(frames_out_v, dim=1))

    return torch.cat(out_k_chunks, dim=1), torch.cat(out_v_chunks, dim=1)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        retrieval_kv=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            retrieval_kv(dict or None): {'k': [B, R, H, D], 'v': [B, R, H, D]} retrieved from bank
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        current_end = current_start + roped_query.shape[1]
        sink_tokens = self.sink_size * frame_seqlen

        # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = roped_query.shape[1]
        if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
            # Calculate the number of new tokens added in this step
            # Shift existing cache content left to discard oldest tokens
            # Clone the source slice to avoid overlapping memory error
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            # Insert the new keys/values at the end
            local_end_index = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            # Assign new keys/values directly up to current_end
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v

        # Compose attention window: [sink | retrieval | recent] or full cache
        if retrieval_kv is not None:
            sink_k = kv_cache["k"][:, :sink_tokens]
            sink_v = kv_cache["v"][:, :sink_tokens]
            retr_tokens = retrieval_kv['k'].shape[1]
            recent_budget = max(0, max_attention_size - sink_tokens - retr_tokens)
            recent_k = kv_cache["k"][:, max(sink_tokens, local_end_index - recent_budget):local_end_index]
            recent_v = kv_cache["v"][:, max(sink_tokens, local_end_index - recent_budget):local_end_index]
            retr_k = retrieval_kv['k'].to(roped_query.dtype).clone()
            retr_v = retrieval_kv['v'].to(roped_query.dtype)

            # RoPE correction: shift retrieval keys to be contiguous with recent
            if retrieval_kv.get('rope_correction', False) and 'src_frame_ids' in retrieval_kv:
                current_end_frame = current_end // frame_seqlen
                recent_len_frames = recent_k.shape[1] // frame_seqlen
                recent_start_frame = current_end_frame - recent_len_frames
                src_frame_ids = retrieval_kv['src_frame_ids']  # list of original frame start per chunk
                retr_num_chunks = len(src_frame_ids)
                # Per-chunk token stride: with store-time compression, compressed
                # chunks are not multiples of frame_seqlen, so use the actual stride.
                chunk_tokens = retr_tokens // retr_num_chunks
                chunk_size_frames = retrieval_kv.get(
                    'compress_chunk_size', chunk_tokens // frame_seqlen)
                for ci, src_fid in enumerate(src_frame_ids):
                    # virtual position: right before recent, after sink
                    virt_fid = recent_start_frame - (retr_num_chunks - ci) * chunk_size_frames
                    delta = virt_fid - src_fid
                    if delta != 0:
                        tok_s = ci * chunk_tokens
                        tok_e = tok_s + chunk_tokens
                        _rope_time_delta_mul_(retr_k[:, tok_s:tok_e], freqs, delta)

            if retrieval_kv.get('compress', False):
                retr_k, retr_v = _compress_retrieval_kv(
                    retr_k, retr_v,
                    chunk_size=retrieval_kv['compress_chunk_size'],
                    frame_seqlen=frame_seqlen,
                    keep_ratio=retrieval_kv['compress_keep_ratio'],
                    anchor_rotate=retrieval_kv['compress_anchor_rotate'],
                    sp_reduce=False,
                )

            k_cache = torch.cat([sink_k, retr_k, recent_k], dim=1)
            v_cache = torch.cat([sink_v, retr_v, recent_v], dim=1)
        else:
            k_cache = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
            v_cache = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]

        x = attention(roped_query, k_cache, v_cache)

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        
        if crossattn_cache is not None:
            if not crossattn_cache.get("is_init", False):
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim=dim, 
                                                num_heads=num_heads, 
                                                local_attn_size=local_attn_size, 
                                                sink_size=sink_size, 
                                                qk_norm=qk_norm, 
                                                eps=eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        retrieval_kv=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32
        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs, kv_cache, current_start, max_attention_size,
            retrieval_kv=retrieval_kv)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cam injection (only if dit_cond_dict is provided and contains c2ws_plucker_emb)
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            c2ws_hidden_states = self.cam_injector_layer2(torch_F.silu(self.cam_injector_layer1(c2ws_plucker_emb)))
            c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb
            cam_scale = self.cam_scale_layer(c2ws_hidden_states)
            cam_shift = self.cam_shift_layer(c2ws_hidden_states)
            x = (1.0 + cam_scale) * x + cam_shift

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, 
                                    crossattn_cache=crossattn_cache)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModelFast(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 control_type='cam',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=21,
                 sink_size=3,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            control_type (`str`, *optional*, defaults to 'cam'):
               Type of conditioning control signal - 'cam' (6-dim camera Plucker
               embeddings) or 'act' (7-dim action embeddings including WASD movement)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        if control_type == 'cam':
            control_dim = 6
        elif control_type == 'act':
            control_dim = 7

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        
        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim * 64 * patch_size[0] * patch_size[1] * patch_size[2], dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)
        
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(dim, ffn_dim, num_heads, 
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        retrieval_kv=None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            dit_cond_dict (`dict`, *optional*, defaults to None):
                Dictionary of conditioning signals. May contain key ``c2ws_plucker_emb``
                with camera Plucker embeddings of shape [B, C, F, H, W] for camera control.
            kv_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer self-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, kv_size, num_heads, head_dim]), ``global_end_index``,
                and ``local_end_index`` (scalar Tensors tracking cache position).
            crossattn_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer cross-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, text_len, num_heads, head_dim]) and ``is_init`` (bool).
            current_start (`int`, *optional*, defaults to 0):
                Token offset of the current chunk in the full sequence. Used to index
                into the KV cache and compute positional embeddings correctly.
            max_attention_size (`int`, *optional*, defaults to 1_000_000):
                Maximum number of KV tokens each query can attend to. Limits the
                effective context window of self-attention to control memory usage.

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert y is not None
        
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_lens)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_lens)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # cam
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            c2ws_plucker_emb = [
                rearrange(
                    i,
                    '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                    c1=self.patch_size[0],
                    c2=self.patch_size[1],
                    c3=self.patch_size[2],
                ) for i in c2ws_plucker_emb
            ]
            c2ws_plucker_emb = torch.cat(
                c2ws_plucker_emb, dim=1)  # [1, (L1+...+Ln), C]
            c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
            c2ws_hidden_states = self.c2ws_hidden_states_layer2(
                torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb)))
            dit_cond_dict = dict(dit_cond_dict)
            dit_cond_dict["c2ws_plucker_emb"] = (
                c2ws_plucker_emb + c2ws_hidden_states)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dit_cond_dict=dit_cond_dict,
            max_attention_size=max_attention_size)

        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
                    "retrieval_kv": retrieval_kv[block_index] if retrieval_kv else None,
                }
            )
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        return [u.float() for u in x]


    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

