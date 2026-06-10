#!/usr/bin/env python3
"""Eval script for LoRA adapter checkpoints (PEFT save_pretrained format)."""
import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dataset_optimized import build_wds_from_args
from src.data.normal_datamodule_optimized import NormalDataModule
from scripts.eval_metrics_DF import compute_eer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", required=True, help="Path to PEFT adapter dir (epoch_NNN.ckpt/)")
    p.add_argument("--base_ckpt", default="/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--protocol_path", required=True, help="Eval protocol (test subset)")
    p.add_argument("--wds_data_dir", default=None)
    p.add_argument("--trim_length", type=int, default=32000)
    p.add_argument("--padding_type", default="repeat", choices=["repeat", "zero"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ssl_pretrained_path", default="pretrained/xlsr2_300m.pt")
    p.add_argument("--tag", default="", help="Label for this run in report")
    return p.parse_args()


def load_model(args):
    from src.models.components.xlsr_conformertcm import Model as XLSRConformerTCM
    from peft import PeftModel

    conformer_args = dict(emb_size=144, heads=4, kernel_size=31, n_encoders=4,
                          type="conv", pooling="first")

    print(f"Loading base model from {args.ssl_pretrained_path}")
    net = XLSRConformerTCM(conformer_args, args.ssl_pretrained_path)

    print(f"Loading MDT weights from {args.base_ckpt}")
    ckpt = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    ckpt = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)

    print(f"Loading LoRA adapter from {args.adapter_path}")
    net = PeftModel.from_pretrained(net, args.adapter_path)
    net.eval()
    return net.to(args.device)


class ProtocolDataset(torch.utils.data.Dataset):
    LABEL_MAP = {"bonafide": 1, "spoof": 0}

    def __init__(self, data_dir, protocol_path, trim_length, padding_type):
        import torchaudio
        from src.data.components.dataio import pad_tensor
        self.data_dir = data_dir
        self.trim_length = trim_length
        self.padding_type = padding_type
        self.torchaudio = torchaudio
        self.pad_tensor = pad_tensor
        self.samples = []
        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().rsplit(maxsplit=2)
                if len(parts) == 3:
                    path, subset, label = parts
                    self.samples.append((path, self.LABEL_MAP.get(label.lower(), 0)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        wav_path = os.path.join(self.data_dir, rel_path)
        wav, sr = self.torchaudio.load(wav_path)
        wav = wav.squeeze(0).float()
        if sr != 16000:
            wav = self.torchaudio.functional.resample(wav, sr, 16000)
        wav = self.pad_tensor(wav, padding_type=self.padding_type,
                              max_len=self.trim_length, random_start=False)
        return wav, label


def run_eval(model, data_args, args):
    from torch.utils.data import DataLoader

    print(f"Building file-based eval dataset from {args.protocol_path}")
    ds = ProtocolDataset(args.data_dir, args.protocol_path,
                         args.trim_length, args.padding_type)
    print(f"  {len(ds)} eval samples")
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=False)

    all_scores, all_labels = [], []
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device)
            logits = model(x)
            scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_scores.append(scores)
            all_labels.append(np.array(y))
            n += len(scores)
            if n % 10000 == 0:
                print(f"  processed {n}/{len(ds)} samples...")

    print(f"  done: {n} samples")
    return np.concatenate(all_scores), np.concatenate(all_labels)


def report(scores, labels, tag=""):
    bonafide = scores[labels == 1]
    spoof    = scores[labels == 0]

    eer, eer_thresh = compute_eer(bonafide, spoof)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, scores)

    preds = (scores >= eer_thresh).astype(int)
    acc = (preds == labels).mean()

    header = f"=== {tag} ===" if tag else "=== Results ==="
    print(header)
    print(f"  EER      : {eer*100:.2f}%  (thresh={eer_thresh:.4f})")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Acc@EER  : {acc*100:.2f}%")
    print(f"  N bonafide={len(bonafide)}  N spoof={len(spoof)}")
    return dict(eer=eer, auc=auc, acc=acc)


def main():
    args = parse_args()
    model = load_model(args)

    wds_data_dir = args.wds_data_dir or os.path.join(args.data_dir, "wds_data")
    data_args = {
        "wds_data_dir": wds_data_dir,
        "data_dir": args.data_dir,
        "protocol_path": args.protocol_path,
        "trim_length": args.trim_length,
        "padding_type": args.padding_type,
        "random_start": False,
        "wav_samp_rate": 16000,
        "wds_use_brace_pattern": True,
        "wds_shard_shuffle": 0,
        "wds_sample_shuffle": 0,
        "wds_eval_return_utt_id": True,
        "augmentation_methods": ["none"],
    }

    scores, labels = run_eval(model, data_args, args)
    report(scores, labels, tag=args.tag or args.adapter_path)


if __name__ == "__main__":
    main()
