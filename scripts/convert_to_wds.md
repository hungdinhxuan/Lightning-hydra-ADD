
```bash
python3 scripts/convert_to_wds.py \
  --input_wav_dir data/HABLA_novel_sets/FinalDataset_16khz \
  --protocol_path data/HABLA_novel_sets/FinalDataset_16khz/protocol.txt \
  --output_dir /dev/shm/HABLA_novel_sets \
  --shard_size_mb 1024 --use_dev_shm
```