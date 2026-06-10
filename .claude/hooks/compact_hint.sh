#!/usr/bin/env bash

cat <<'EOF'
[COMPACT INSTRUCTIONS]
Keep:
- files changed and why
- active bug hypotheses
- commands that worked
- benchmark/warm-up summaries
- chosen optimized config from logs/optimized_configs/

Drop:
- long train logs
- repeated stack traces
- large directory listings
- repeated grep output
EOF