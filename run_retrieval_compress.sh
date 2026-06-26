#!/usr/bin/env bash
set -euo pipefail
set -x

WEIGHT_DIR=./lingbot-world-base-cam
PROMPTS_JSON=./examples/new/prompts.json
IMAGE_DIR=./examples/new

TRAJ_DIRS=(
  ./traj/R_L_R_L_R_L90_360
)


CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
SEED="${SEED:-42}"

# Retrieval frames: 0, 3, 6, 12, 18, 24, 30
for retr_frames in 18; do
  SAVE_DIR="./output"

  for sample in $(seq -w 00 00); do
    image_path=""
    for ext in png jpg jpeg webp; do
      if [[ -f "${IMAGE_DIR}/${sample}.${ext}" ]]; then
        image_path="${IMAGE_DIR}/${sample}.${ext}"
        break
      fi
    done
    if [[ -z "${image_path}" ]]; then
      echo "Skip: missing image ${IMAGE_DIR}/${sample}.*"
      continue
    fi

    prompt="$(
      python - "${PROMPTS_JSON}" "${sample}" <<'PY'
import json
import sys

json_path, sample_id = sys.argv[1], sys.argv[2]
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if sample_id in data:
    print(data[sample_id])
else:
    for ext in ("png", "jpg", "jpeg", "webp"):
        key = f"{sample_id}.{ext}"
        if key in data:
            print(data[key])
            break
    else:
        raise KeyError(sample_id)
PY
    )"

    for traj_dir in "${TRAJ_DIRS[@]}"; do
      traj_name="$(basename "${traj_dir}")"
      frame_num="${traj_name##*_}"

      traj_stem="${traj_name%_*}"
      expected_output="${SAVE_DIR}/${sample}_${traj_stem}_${frame_num}.mp4"
      if [[ -f "${expected_output}" ]]; then
        echo "Skip: ${expected_output} already exists"
        continue
      fi

      retrieval_args=()
      if [[ "${retr_frames}" -gt 0 ]]; then
        retrieval_args=(
          --use_retrieval
          --retrieval_frames "${retr_frames}"
          --kv_compression_enable
          --kv_compression_keep_ratio "${KEEP_RATIO}"
          --kv_compression_at_store
        )
      fi

      CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --nproc_per_node="${NPROC_PER_NODE}" generate_fast.py \
        --task i2v-A14B \
        --size 480*832 \
        --ckpt_dir "${WEIGHT_DIR}" \
        --image "${image_path}" \
        --action_path "${traj_dir}" \
        --dit_fsdp \
        --t5_fsdp \
        --ulysses_size "${NPROC_PER_NODE}" \
        --frame_num "${frame_num}" \
        --prompt "${prompt}" \
        --save_dir "${SAVE_DIR}" \
        "${retrieval_args[@]}" \
        --kv_bank_on_gpu \
        --kv_compression_pooled
    done
  done
done
