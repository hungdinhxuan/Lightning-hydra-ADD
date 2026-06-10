#!/usr/bin/env bash

cat <<'EOF'
[AUDIO-DF REPO MAP]
- Prefer reading: src/, configs/, scripts/, tests/
- Avoid broad scans in: data/, runs/, logs/results/, checkpoints/, .git/
- logs/optimized_configs/ stores approved benchmark/warm-up summaries
- For data loading changes, prefer warm-up benchmark before full training
- Return metric summaries first: EER, AUC, minDCF, accuracy, threshold, OOM, Traceback
EOF