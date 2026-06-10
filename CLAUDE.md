# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Audio deepfake detection (ADD) built on the lightning-hydra-template. Core model: XLSR (wav2vec2 SSL via fairseq) → Conformer-TCM backend, with MDT (multi-duration-truncation) multi-view training and LoRA adapter fine-tuning (PEFT). Data is loaded through WebDataset shards for DDP training.

## Commands

```bash
# Environment (uv-based; Python pinned to 3.9.21 in pyproject.toml)
uv sync
# fairseq must be installed editable from the vendored copy:
cd fairseq_lib && TORCH_CUDA_VERSION=cu121 FORCE_CUDA=1 pip install -e .

# Train (Hydra entry point)
python src/train.py experiment=xlsr_conformertcm_mdt_lora_ultrashort_vadshort
python src/train.py trainer.max_epochs=20 data.batch_size=64   # any param overridable
# Full DDP + WebDataset pipeline (preflight, shard conversion, launch):
scripts/train_dsd_ddp_optimized.sh

# Evaluate a checkpoint
python src/eval.py ckpt_path=<path>
# Standalone LoRA adapter eval (EER/AUC/acc @ EER threshold):
python scripts/eval_lora_adapter.py --adapter_path <ckpt_dir> --base_ckpt <pt> --data_dir <dir> --protocol_path <txt>

# Tests
pytest                          # all (or: make test-full)
make test                       # skips tests marked slow
pytest tests/test_wds_keys.py -v          # single file
pytest tests/test_extract_generators.py::test_name -v   # single test

# Lint/format
make format                     # pre-commit run -a
make clean                      # remove caches/artifacts
```

## Required setup

- `.project-root` marker is used by `rootutils.setup_root()` in `src/train.py` / `src/eval.py` — it adds the repo to PYTHONPATH, sets `PROJECT_ROOT` (referenced in config paths), and loads `.env`.
- Copy `.env.example` → `.env`. Required: `XLSR_PRETRAINED_MODEL_PATH`, `NOISE_DATASET_PATH`, `NOISE_DATASET_PROTOCOL`, `WDS_DATA_DIR`.
- XLSR2 300M checkpoint expected at `pretrained/xlsr2_300m.pt` (auto-fetched from S3 via boto3 if missing — see `src/models/components/xlsr_conformertcm.py`).

## Architecture

### LightningModule hierarchy (src/models/)

```
BaseLitModule (base/base_module.py)        — metrics, buffered/async score writing, abstract init_model()/init_criteria()
└── AdapterLitModule (base/adapter_module.py) — loads base ckpt (local/S3/MLflow), applies PEFT LoRA config
    └── MDTLitModule (base/mdt_module.py)      — multi-view (multi-duration) training, per-view accuracy, adaptive view weights
```

Concrete modules (`xlsr_conformertcm_MDT_module.py`, `xlsr_conformertcm_normal_module.py`, `aasist_normal_module.py`) plug architectures from `src/models/components/` (`xlsr_conformertcm.py` wraps fairseq XLSR + custom Conformer; `aasist.py`).

### Data pipeline (src/data/)

- DataModules: `normal_datamodule_optimized.py` and `normal_MDT_datamodule_optimized.py` — WebDataset-based, built via `build_wds_from_args()` in `dataset_optimized.py`.
- Protocol files: lines of `relative/path.wav subset label` with subsets `train`/`dev`/`eval`.
- `wds_keys.py` maps file paths to tar-safe member keys.
- MDT views are sample lengths at 16 kHz (e.g. `[8000, 16000, 24000, 32000]` = 0.5–2.0 s); padding `repeat` or `zero`; `random_start` true for train, false for eval.
- Augmentation (RawBoost etc.) in `src/data/components/augwrapper.py`.

### Hydra configs (configs/)

`train.yaml` composes defaults: data → model → callbacks → logger → trainer → paths → experiment (optional, overrides several groups at once) → optional `local/` (machine-specific, gitignored) → debug. Experiments live in `configs/experiment/` (e.g. `xlsr_conformertcm_mdt_lora_ultrashort_vadshort.yaml`) and reference model configs like `xlsr_conformertcm_MDT_LoRA.yaml`.

### Metrics & evaluation

EER/AUC/minDCF are computed post-hoc from score files, not in-model — see `scripts/eval_metrics_DF.py` (`compute_eer`). `scripts/benchmark_py/short_duration_eval.py` runs the 720k short-duration inference matrix.

## Critical gotchas

- **LoRA + DDP**: must use `++trainer.strategy=ddp_find_unused_parameters_true` — LoRA only targets some modules (e.g. fc1/fc2) and unused params hang plain DDP.
- **WebDataset + DDP epoch sizing**: set `wds_auto_epoch_batches: true` (in experiment data config) to avoid shard-exhaustion hangs; `train_dsd_ddp_optimized.sh` also validates `TRAIN_SHARDS >= GPU_COUNT * NUM_WORKERS`.
- **Before changing data loading**, run the warm-up benchmark workflow in `OPTIMIZE_TRAINING_SPEED.md` (preflight → shard conversion → data-only benchmark → micro-train) rather than launching a full training run. Approved benchmark summaries live in `logs/optimized_configs/`.
- Post-training checkpoint averaging available via `++model_averaging=true` (see `src/train.py`).
- Avoid scanning `data/`, `runs/`, `logs/results/`, checkpoint dirs — very large.

## Reporting conventions

When summarizing training/eval results, lead with metrics: EER, AUC, minDCF, accuracy, threshold; surface OOM errors and tracebacks verbatim.

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
