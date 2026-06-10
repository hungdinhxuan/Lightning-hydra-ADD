#!/usr/bin/env python3
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool_name = payload.get("tool_name", "")
tool_input = payload.get("tool_input", {}) or {}

def deny(reason: str):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason
        }
    }))
    sys.exit(0)

def context(msg: str):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": msg
        }
    }))
    sys.exit(0)

large_paths = ["data/", "runs/", "logs/results/", "checkpoints/", ".git/"]

if tool_name == "Bash":
    cmd = tool_input.get("command", "")

    if any(x in cmd for x in ["find .", "find ..", "find ~", "find /"]) and any(p in cmd for p in large_paths):
        deny("Broad find over large/noisy directories is blocked. Search src/, configs/, scripts/, or tests/ instead.")

    if any(x in cmd for x in ["grep -R", "rg ", "ripgrep "]) and any(p in cmd for p in large_paths):
        deny("Broad recursive search in large artifact directories is blocked. Narrow the path first.")

    if "pytest" in cmd:
        context("Prefer compact pytest output, focused on FAIL/ERROR and short tails.")

    if any(x in cmd for x in ["benchmark", "warmup"]):
        context("For benchmark/warm-up commands, prefer metric summary first: EER/AUC/minDCF/accuracy/threshold, then warnings/errors.")

    if "train.py" in cmd:
        context("Before long training runs, prefer config validation, smoke run, or warm-up benchmark.")

elif tool_name == "Read":
    path = tool_input.get("file_path", "")
    if any(p in path for p in large_paths):
        context(f"Reading large artifact path: {path}. Prefer summary files, metadata, or narrower files first.")

elif tool_name in ("Edit", "Write"):
    path = tool_input.get("file_path", "")
    if "logs/optimized_configs/" in path:
        context("Preserve markdown structure in logs/optimized_configs; treat it as experiment record, not scratch output.")