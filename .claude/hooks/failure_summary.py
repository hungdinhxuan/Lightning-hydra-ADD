#!/usr/bin/env python3
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)

text = json.dumps(payload, ensure_ascii=False)

labels = []
for key in [
    "CUDA out of memory",
    "Traceback",
    "ModuleNotFoundError",
    "KeyError",
    "FileNotFoundError",
    "Permission denied",
    "Hydra",
    "No such file or directory"
]:
    if key.lower() in text.lower():
        labels.append(key)

if labels:
    print("Failure summary: " + ", ".join(labels[:4]))