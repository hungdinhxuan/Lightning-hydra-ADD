---
name: add-ultra-short-eval
description: Prepare and run the 720k short-duration inference matrix (full-length + fixed-length, random_start on/off, 4 durations) for an ADD model checkpoint. Use when the user wants to evaluate a trained MDT/LoRA ultrashort model, run the protocol_test_eval benchmark, score observed_short/vad_short tracks, or produce the score files a trend report consumes. Triggers: "run inference matrix", "720k eval", "short_duration_eval prepare", "benchmark lora checkpoint", "fixed-length vs full-length eval".
---

# ADD Short-Duration Inference Matrix (720k)

Score a checkpoint across the full short-duration protocol: full-length tracks + a fixed-length matrix (2 random_start modes × 4 durations).

## When to use
- Evaluate a finished checkpoint (e.g. from `add-lora-train`).
- Regenerate score files feeding `add-trend-report`.

## Matrix composition (720,000 rows)
```
full-length:   observed_short_full 5,255 + vad_short_full 74,745  = 80,000
fixed-length:  1 track × 2 random_start × 4 durations × 80,000     = 640,000
durations:     0.5s, 1.0s, 1.5s, 2.0s     (samples 8000/16000/24000/32000)
```

## Protocol distinction (critical)
- **Full-length** uses processed short audio directly: `observed_short_full`, `vad_short_full`.
- **Fixed-length** maps every sample back to its original source utterance, then crops:
  `audio/original_utterance/<source_stem>__source.wav`.

## Score file format
```
<filename> <spoof_score> <bonafide_score>
```
Primary metric uses `bonafide_score`. Higher = more bonafide.

## 1. Prepare benchmark work dir
`scripts/benchmark_py/short_duration_eval.py` subcommands: `validate prepare enrich-single aggregate report run-single`.
```bash
python scripts/benchmark_py/short_duration_eval.py prepare \
  --source-root data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026 \
  --work-dir data/protocol_test_eval_benchmark_lora_full \
  --subsets test \
  --tracks fixed_length_test observed_short_full vad_short_full \
  --comment-prefix lora_ultrashort_mdt_ddp_full100_ep21
```
Prepared tracks: `fixed_length_test` 80,000 / `observed_short_full` 5,255 / `vad_short_full` 74,745.

## 2. Run the matrix (wrapper)
Wrapper drives raw-audio eval (NOT WDS): `data=MDT_default`, `++data.args.padding_type=repeat`, `++data.args.view_lengths_samples=[8000,16000,24000,32000]`.
```bash
cd /home/hungdx/code/Lightning-hydra-ADD
export REPO_ROOT="$PWD"
export BASE_CKPT="/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt"
export LORA_CKPT="$REPO_ROOT/logs/train/runs/2026-05-30_19-32-38/checkpoints/epoch_021.ckpt"
export WORK_DIR="data/protocol_test_eval_benchmark_lora_full"
export RESULTS_ROOT="logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch"
export PREFIX="lora_ultrashort_mdt_ddp_full100_ep21"

CUDA_VISIBLE_DEVICES=0 \
  bash scripts/cnsl/May302026/eval_lora_ultrashort_vadshort.sh -d 0 \
  2>&1 | tee logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch.log
```
Lightning loads `++model.base_model_path=$BASE_CKPT` then `ckpt_path=$LORA_CKPT`.

## Output layout
- Results root: `logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch`
- Fixed-length filename: `fixed_length_test_xlsr_conformertcm_mdt_lora_ultrashort_vadshort_<PREFIX>_{norandom|random}_{duration}.txt`
- Full-length under `<PREFIX>_full/`: `{observed_short_full|vad_short_full}_..._<PREFIX>_full.txt`

## Verify
```bash
ps -eo pid,cmd | rg 'short_duration_eval|eval_lora_ultrashort' | rg -v 'rg ' || true
find logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch -name '*.txt' | wc -l
```
Report metric summary first: EER/AUC/accuracy/threshold, then warnings/errors.

## Gotchas
- Fixed-length uses original source utterances, not pre-shortened segments.
- Keep `<PREFIX>` identical to the prepare step or downstream report globbing breaks.

Next stage: skill `add-compare-ultra-short-report`.
