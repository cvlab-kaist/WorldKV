<div align="center">
<h1>WorldKV: Efficient World Memory with World Retrieval and Compression</h1>

[**Jung Yi**](https://yj-142150.github.io/jungyi/)<sup>1</sup> · [**Minjae Kim**](https://github.com/kwjames98/)<sup>1</sup> · [**Paul Hyunbin Cho**](https://openreview.net/profile?id=~Paul_Hyunbin_Cho1)<sup>1</sup> · [**Wooseok Jang**](https://woo1726.github.io/)<sup>1</sup> · [**Sangdoo Yun**](https://sangdooyun.github.io/)<sup>2</sup> · [**Seungryong Kim**](https://cvlab.korea.ac.kr)<sup>1</sup>

<sup>1</sup>KAIST AI&emsp;<sup>2</sup>Naver AI

<span style="font-size: 16px; font-weight: 700;">
  <a href="https://cvlab-kaist.github.io/WorldKV/">https://cvlab-kaist.github.io/WorldKV/</a>
</span>

</div>


<div align="center">


</div>

-----
WorldKV is a training-free framework that enables efficient world memory in autoregressive video world models by combining World Retrieval and World Compression.

## 🚀 Progress

- [x] LingBot-World-Fast
- [ ] Inspatio-World
- [ ] Matrix-Game-2.0

### Installation
Clone the repo:
```sh
git clone https://github.com/cvlab-kaist/WorldKV.git
cd WorldKV
```
Install dependencies:
```sh
# Ensure torch >= 2.4.0
pip install -r requirements.txt
```
Install [`flash_attn`](https://github.com/Dao-AILab/flash-attention):
```sh
pip install flash-attn --no-build-isolation
```


Download models using huggingface-cli:
```sh
pip install "huggingface_hub[cli]"

huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./lingbot-world-base-cam

huggingface-cli download robbyant/lingbot-world-fast --local-dir ./lingbot-world-base-cam/lingbot_world_fast
```

### Inference
Before running inference, you need to prepare:
- Input image
- Text prompt
- Control signals 
  - `intrinsics.npy`: Shape `[num_frames, 4]`, where the 4 values represent `[fx, fy, cx, cy]`
  - `poses.npy`: Shape `[num_frames, 4, 4]`, where each `[4, 4]` represents a transformation matrix in OpenCV coordinates

#### Quick Start

A single run on 4 GPUs with camera-pose retrieval + store-time compression:

```sh
CUDA_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 generate_fast.py \
    --task i2v-A14B \
    --size 480*832 \
    --ckpt_dir ./lingbot-world-base-cam \
    --image examples/new/00.png \
    --prompt "A sun-drenched stable courtyard paved with weathered cobblestones stretches before a sturdy stone-and-timber barn, where a chestnut horse stands quietly in the shade — a young adventurer in a blue tunic faces the open structure, shield on their back, as vast rolling green hills and rocky outcrops unfurl under a bright clear sky, evoking the boundless wonder of an open-world journey." \
    --action_path traj/R_L_R_L_R_L90_360 \
    --frame_num 360 \
    --save_dir outputs/demo \
    --dit_fsdp --t5_fsdp --ulysses_size 4 \
    --use_retrieval --retrieval_frames 18 --kv_bank_on_gpu \
    --kv_compression_enable --kv_compression_keep_ratio 0.5 --kv_compression_at_store --kv_compression_pooled
```

> Set `--ulysses_size` equal to `--nproc_per_node` (number of GPUs), and `--frame_num` to the trajectory length (e.g. `traj/..._360` → `360`).

#### Arguments

**Parallelism & memory**

| Argument | Description |
|----------|-------------|
| `offload_model` | Offload model to CPU between forwards. |
| `kv_bank_on_gpu` | Keep the KV bank in VRAM instead of CPU (faster, Recommended when VRAM is enough). |

**KV-cache mode** — mutually exclusive; default is a local `sink + recent` window.

| Argument | Description |
|----------|-------------|
| `use_retrieval` | Retrieve past chunks by camera-pose similarity. Window = `sink + retrieved + recent`. |
| `retrieval_frames` | Frames retrieved from the bank; multiple of `chunk_size` (3), e.g. `18` = 6 chunks. |
| `retrieval_rope_correction` | Rebase retrieved KV to contiguous temporal positions before `recent`. |

**KV compression** — anchor + novelty token pruning on retrieved KV.

| Argument | Description |
|----------|-------------|
| `kv_compression_enable` | Switch for token pruning (required for the flags below). |
| `kv_compression_keep_ratio` | Portion of tokens kept per non-anchor frame (anchor frame kept in full) in each chunk. |
| `kv_compression_at_store` | Prune once at store time. Recommended. |
| `kv_compression_pooled` | Share the keep budget across a chunk's non-anchor frames. |


To run it directly, use the provided script:
``` sh
bash run_retrieval_compress.sh
```


## 📚 Related Projects
- [Deep Forcing](https://cvlab-kaist.github.io/DeepForcing/)
- [LingBot-World](https://github.com/Robbyant/lingbot-world/)
- [Inspatio-World](https://inspatio.github.io/inspatio-world/)
- [Matrix-Game](https://matrix-game-v2.github.io/)

## 📜 License
This project is licensed under the Apache 2.0 License. Please refer to the [LICENSE file](LICENSE.txt) for the full text, including details on rights and restrictions.

## 📖 Citation
If you find this work useful for your research, please cite our paper:

```
@article{yi2026worldkv,
  title={WorldKV: Efficient World Memory with World Retrieval and Compression},
  author={Yi, Jung and Kim, Minjae and Cho, Paul Hyunbin and Jang, Wooseok and Yun, Sangdoo and Kim, Seungryong},
  journal={arXiv preprint arXiv:2605.22718},
  year={2026}
}
```
