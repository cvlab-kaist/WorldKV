import math
import torch
import torch.nn.functional as torch_F
from einops import rearrange

from ..modules.attention import flash_attention
from ..modules.model import sinusoidal_embedding_1d
from ..modules.model_fast import _compress_retrieval_kv
from .ulysses import distributed_attention
from .util import gather_forward, get_rank, get_world_size


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def _rope_time_delta_mul_(k_chunk, freqs, delta_frames):
    """In-place time-axis RoPE shift for already-roped keys."""
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


@torch.amp.autocast('cuda', enabled=False)
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


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    dit_cond_dict=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
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
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
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
        c2ws_plucker_emb = torch.cat(c2ws_plucker_emb,
                                     dim=1)  # [1, (L1+...+Ln), C]
        c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
        c2ws_hidden_states = self.c2ws_hidden_states_layer2(
            torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb)))
        c2ws_plucker_emb = c2ws_plucker_emb + c2ws_hidden_states

        cam_len = c2ws_plucker_emb.size(1)
        if cam_len < seq_len:
            pad_len = seq_len - cam_len
            pad = c2ws_plucker_emb.new_zeros(
                c2ws_plucker_emb.size(0), pad_len, c2ws_plucker_emb.size(2))
            c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, pad], dim=1)
        elif cam_len > seq_len:
            c2ws_plucker_emb = c2ws_plucker_emb[:, :seq_len, :]

        if get_world_size() > 1:
            c2ws_plucker_emb = torch.chunk(
                c2ws_plucker_emb, get_world_size(), dim=1)[get_rank()]
        dit_cond_dict = dict(dit_cond_dict)
        dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        dit_cond_dict=dit_cond_dict)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


def sp_dit_forward_causal(
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
    """
    x:                  A list of videos each with shape [C, T, H, W].
    t:                  [B].
    context:            A list of text embeddings each with shape [L, C].
    seq_len:            Maximum sequence length for positional encoding.
    y:                  Conditional video inputs for image-to-video mode, same shape as x.
    dit_cond_dict:      Dictionary of conditioning signals. May contain key ``c2ws_plucker_emb``
                        with camera Plucker embeddings of shape [B, C, F, H, W] for camera control.
    kv_cache:           Per-layer self-attention KV cache. Each dict contains keys ``k``, ``v``
                        (Tensor of shape [B, kv_size, local_heads, head_dim]), ``global_end_index``,
                        and ``local_end_index`` (scalar Tensors tracking cache position).
    crossattn_cache:    Per-layer cross-attention KV cache. Each dict contains keys ``k``, ``v``
                        (Tensor of shape [B, text_len, num_heads, head_dim]) and ``is_init`` (bool).
    current_start:      Token offset of the current chunk in the full sequence. Used to index
                        into the KV cache and compute positional embeddings correctly.
    max_attention_size: Maximum number of KV tokens each query can attend to. Limits the
                        effective context window of self-attention to control memory usage.

    Note: This function uses pure Ulysses attention (head splitting only).
    Each GPU processes the full sequence but only a subset of attention heads.
    Hence, it is the same with forward method in model_fast.py.
    """

    assert len(x) == 1

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
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)[0]
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
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
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
        c2ws_plucker_emb = torch.cat(c2ws_plucker_emb,
                                     dim=1)  # [1, (L1+...+Ln), C]
        c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
        c2ws_hidden_states = self.c2ws_hidden_states_layer2(
            torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb)))
        c2ws_plucker_emb = c2ws_plucker_emb + c2ws_hidden_states

        cam_len = c2ws_plucker_emb.size(1)
        if cam_len > seq_lens:
            c2ws_plucker_emb = c2ws_plucker_emb[:, :seq_lens, :]

        dit_cond_dict = dict(dit_cond_dict)
        dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb

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


def sp_attn_forward_causal(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    kv_cache=None,
    current_start=0,
    max_attention_size=1_000_000,
    retrieval_kv=None):
    r"""
    Sequence-parallel causal self-attention using Ulysses head splitting.

    Args:
        x(Tensor):              Shape [B, L, C]
        seq_lens(Tensor):       Number of valid tokens per sample.
        grid_sizes(Tensor):     Shape [B, 3], the second dimension contains (F, H, W).
        freqs(Tensor):          Rope freqs, shape [1024, C / num_heads / 2].
        kv_cache(dict):         Self-attention KV cache. Contains keys ``k``, ``v``
                                (Tensor of shape [B, kv_size, local_heads, head_dim]),
                                ``global_end_index``, and ``local_end_index``
                                (scalar Tensors tracking cache position).
        current_start(int):     Token offset of the current chunk in the full sequence.
                                Used to index into the KV cache and compute positional
                                embeddings correctly.
        max_attention_size(int): Maximum number of KV tokens each query can attend to.
                                Limits the effective context window of self-attention
                                to control memory usage.
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    sp_size = get_world_size()
    rank = get_rank()
    local_heads = n // sp_size

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    frame_seqlen = math.prod(grid_sizes[0][1:]).item()
    current_start_frame = current_start // frame_seqlen

    q_local = q[:, :, rank * local_heads:(rank + 1) * local_heads, :]  # [B, s, local_heads, d]
    k_local = k[:, :, rank * local_heads:(rank + 1) * local_heads, :]
    v_local = v[:, :, rank * local_heads:(rank + 1) * local_heads, :]

    query = causal_rope_apply(q_local, grid_sizes, freqs, start_frame=current_start_frame).type_as(v_local)
    key   = causal_rope_apply(k_local, grid_sizes, freqs, start_frame=current_start_frame).type_as(v_local)

    current_end = current_start + seq_lens
    sink_tokens = self.sink_size * frame_seqlen
    kv_cache_size = kv_cache["k"].shape[1]
    if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
            seq_lens + kv_cache["local_end_index"].item() > kv_cache_size):
        # Calculate the number of new tokens added in this step
        # Shift existing cache content left to discard oldest tokens
        # Clone the source slice to avoid overlapping memory error
        num_evicted_tokens = seq_lens + kv_cache["local_end_index"].item() - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        # Insert the new keys/values at the end
        local_end_index = kv_cache["local_end_index"].item() + current_end - \
            kv_cache["global_end_index"].item() - num_evicted_tokens
        local_start_index = local_end_index - seq_lens
        kv_cache["k"][:, local_start_index:local_end_index] = key
        kv_cache["v"][:, local_start_index:local_end_index] = v_local
    else:
        # Assign new keys/values directly up to current_end
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        local_start_index = local_end_index - seq_lens
        kv_cache["k"][:, local_start_index:local_end_index] = key
        kv_cache["v"][:, local_start_index:local_end_index] = v_local

    # Compose attention window: [sink | retrieval | recent] or full cache
    if retrieval_kv is not None:
        sink_k = kv_cache["k"][:, :sink_tokens]
        sink_v = kv_cache["v"][:, :sink_tokens]
        retr_tokens = retrieval_kv['k'].shape[1]
        recent_budget = max(0, max_attention_size - sink_tokens - retr_tokens)
        recent_k = kv_cache["k"][:, max(sink_tokens, local_end_index - recent_budget):local_end_index]
        recent_v = kv_cache["v"][:, max(sink_tokens, local_end_index - recent_budget):local_end_index]
        retr_k = retrieval_kv['k'].to(query.dtype).clone()
        retr_v = retrieval_kv['v'].to(query.dtype)

        # RoPE correction: shift retrieval keys to be contiguous with recent
        if retrieval_kv.get('rope_correction', False) and 'src_frame_ids' in retrieval_kv:
            current_end_frame = current_end // frame_seqlen
            recent_len_frames = recent_k.shape[1] // frame_seqlen
            recent_start_frame = current_end_frame - recent_len_frames
            src_frame_ids = retrieval_kv['src_frame_ids']
            retr_num_chunks = len(src_frame_ids)
            # Per-chunk token stride: with store-time compression, compressed
            # chunks are not multiples of frame_seqlen, so use the actual stride.
            chunk_tokens = retr_tokens // retr_num_chunks
            chunk_size_frames = retrieval_kv.get(
                'compress_chunk_size', chunk_tokens // frame_seqlen)
            for ci, src_fid in enumerate(src_frame_ids):
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
                sp_reduce=(sp_size > 1),
            )

        k_cache = torch.cat([sink_k, retr_k, recent_k], dim=1)
        v_cache = torch.cat([sink_v, retr_v, recent_v], dim=1)
    else:
        k_cache = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
        v_cache = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]

    # Attention on local heads, full key/value cache for this rank
    x_local = flash_attention(query, k_cache, v_cache)  # [B, s, local_heads, d]

    # Gather all head results across GPUs: [B, s, n, d]
    x_full = gather_forward(x_local, dim=2)  # all_gather on head dim

    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end_index)

    # output
    x_full = x_full.flatten(2)
    x_full = self.o(x_full)
    return x_full
