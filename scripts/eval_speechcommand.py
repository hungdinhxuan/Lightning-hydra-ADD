#!/usr/bin/env python3
"""Score speechcommand (all-bonafide) full-length natural for one model.

Output score file: ``<rel_path> <spoof_logit> <bonafide_logit>`` (raw forward
output, same scale as the trend-report score files). Full-length natural means
batch_size=1, no padding -> padding independent.
"""
import argparse
import os
import warnings

import torch
import torchaudio

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--base_ckpt", default="/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt")
    p.add_argument("--adapter_path", default="", help="PEFT adapter dir; empty = baseline")
    p.add_argument("--data_root", default="/data/speechcommand")
    p.add_argument("--testing_list", default="/data/speechcommand/testing_list.txt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ssl_pretrained_path", default="pretrained/xlsr2_300m.pt")
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_model(args):
    from src.models.components.xlsr_conformertcm import Model as XLSRConformerTCM

    conformer_args = dict(emb_size=144, heads=4, kernel_size=31, n_encoders=4,
                          type="conv", pooling="first")
    print(f"[{args.name}] base XLSR from {args.ssl_pretrained_path}", flush=True)
    net = XLSRConformerTCM(conformer_args, args.ssl_pretrained_path)

    print(f"[{args.name}] MDT weights from {args.base_ckpt}", flush=True)
    ckpt = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    ckpt = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)

    if args.adapter_path:
        from peft import PeftModel
        print(f"[{args.name}] LoRA adapter from {args.adapter_path}", flush=True)
        net = PeftModel.from_pretrained(net, args.adapter_path)
    net.eval()
    return net.to(args.device)


def main():
    args = parse_args()
    model = load_model(args)

    rels = [ln.strip() for ln in open(args.testing_list) if ln.strip()]
    print(f"[{args.name}] {len(rels)} clips -> {args.out}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as fh, torch.no_grad():
        for rel in rels:
            wav, sr = torchaudio.load(os.path.join(args.data_root, rel))
            wav = wav.squeeze(0).float()
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            x = wav.unsqueeze(0).to(args.device)
            out = model(x).detach().float().cpu()[0]
            fh.write(f"{rel} {out[0].item()} {out[1].item()}\n")
            n += 1
            if n % 2000 == 0:
                print(f"[{args.name}]   {n}/{len(rels)}", flush=True)
    print(f"[{args.name}] done {n}", flush=True)


if __name__ == "__main__":
    main()
