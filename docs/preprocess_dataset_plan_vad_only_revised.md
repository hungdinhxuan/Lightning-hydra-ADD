# Kế hoạch pre-process dataset cho US-MDT: VAD-only / observed-short

## 1. Quyết định chốt

Kế hoạch này thay thế bản trước theo hướng **không tạo fixed-window segment**.

Lý do bỏ fixed-window ở giai đoạn pre-process chính:

- Mục tiêu practical hiện tại là làm model robust hơn với **speech-active short segments**, gần với production hơn random crop.
- Fixed-window dễ tạo nhiều đoạn silence / non-speech / partial speech, làm nhiễu training và evaluation nếu chưa kiểm soát kỹ.
- VAD-based segment dễ audit hơn: mỗi file nhỏ sau xử lý đều có speech rõ ràng, có `speech_start`, `speech_end`, `speech_duration`.
- Nếu sau này cần fixed-window để ablation, có thể tạo riêng như một experiment phụ, không đưa vào pipeline chính.

Pipeline chính sẽ gồm:

```text
1. Scan protocol.txt của từng dataset.
2. Resolve rel_path -> absolute audio path.
3. Đọc audio, chuẩn hóa sample rate/channel nếu cần.
4. Chạy Silero VAD hoặc wrapper customize từ:
   ~/code/preprocessors/models/silero_vad.py
5. Tạo speech-active segments theo duration bin.
6. Cắt audio thành file nhỏ thật để dễ audit.
7. Lưu protocol mới với cấu trúc tương tự dataset gốc.
8. Lưu manifest đầy đủ để trace ngược về file gốc.
9. Sinh audit report.
```

---

## 2. Input dataset hiện tại

### 2.1. Benchmark / evaluation data

Root:

```text
/data/add_eval_data
```

Cấu trúc hiện tại:

```text
/data/add_eval_data
├── 2025_Kipot
│   ├── 2025
│   └── protocol.txt
├── german_dataset_May262026
│   ├── protocol.txt
│   ├── test.tar
│   └── tts-output
├── M-AILABS
│   ├── de_DE
│   ├── en_UK
│   ├── en_US
│   ├── es_ES
│   ├── fr_FR
│   ├── it_IT
│   ├── pl_PL
│   ├── protocol.txt
│   ├── ru_RU
│   └── uk_UK
├── MLAAD_dev_v10_April21_2026
│   ├── fake
│   └── protocol.txt
├── MLAAD_v6
│   ├── fake
│   └── protocol.txt
├── MLAAD_v7
│   ├── fake
│   └── protocol.txt
├── MLAAD_v8
│   ├── fake
│   └── protocol.txt
└── MLAAD_v9
    ├── fake
    └── protocol.txt
```

Dự kiến số lượng test/eval:

```text
~1,000,000 utterances
```

### 2.2. Training/dev data

Root:

```text
/data/dsd_corpus_pool_13May2026
```

Cấu trúc hiện tại:

```text
/data/dsd_corpus_pool_13May2026
├── 15_May_full.txt
├── 1-phone_large-corpus
├── 2026_April_Dataset_Jiwon_collected
├── April_Synthesizers
├── AudioSet-Strong-Human-Sounds-enyoukai-hf-full-processed
├── CommonVoice_2026_Maysubset-vad
├── commonvoice26_de_en_ko_4000
├── dsd_corpus_pool_13May2026_15_May_full.files
├── FEB_dataset
├── KlingAI-April-2026
├── K-SASV-Bonafide-Training-May
├── MLAAD_v6
├── MLAAD_v7
└── telephony_dec25
```

Dự kiến số lượng:

```text
train: ~544,750 utterances
dev:   ~100,000 utterances
```

### 2.3. Protocol format gốc

Mỗi `protocol.txt` có format:

```text
<rel_path> <subset> <label>
```

Ví dụ concept:

```text
audio/example_001.wav train bonafide
audio/example_002.wav test spoof
fake/example_003.wav test spoof
```

Quy ước label thống nhất trong pipeline:

```text
bonafide / spoof
```

Nếu gặp label khác, map về:

```text
real, human, genuine -> bonafide
fake, spoof, tts, vc, synthetic -> spoof
```

---

## 3. Output dataset sau pre-process

Nên giữ cấu trúc tương tự dataset gốc, nhưng thêm cấp `vad_short` và duration bin để dễ audit.

Root output đề xuất:

```text
/data/add_processed_vad_short_29May2026
```

Cấu trúc tổng quát:

```text
/data/add_processed_vad_short_29May2026
├── train
│   ├── protocol.txt
│   ├── manifest.parquet
│   ├── manifest.csv
│   ├── audit_report.md
│   └── audio
│       ├── vad_0p5_1p0
│       ├── vad_1p0_1p5
│       ├── vad_1p5_2p0
│       └── observed_short_0p5_2p0
├── dev
│   ├── protocol.txt
│   ├── manifest.parquet
│   ├── manifest.csv
│   ├── audit_report.md
│   └── audio
│       ├── vad_0p5_1p0
│       ├── vad_1p0_1p5
│       ├── vad_1p5_2p0
│       └── observed_short_0p5_2p0
├── test
│   ├── protocol.txt
│   ├── manifest.parquet
│   ├── manifest.csv
│   ├── audit_report.md
│   └── audio
│       ├── vad_0p5_1p0
│       ├── vad_1p0_1p5
│       ├── vad_1p5_2p0
│       └── observed_short_0p5_2p0
├── metadata
│   ├── metadata_raw_all.parquet
│   ├── vad_segments_all.parquet
│   ├── failed_files.csv
│   ├── preprocess_config.yaml
│   └── global_audit_report.md
└── logs
    ├── preprocess_train.log
    ├── preprocess_dev.log
    └── preprocess_test.log
```

`protocol.txt` sau xử lý vẫn giữ format tương tự:

```text
<rel_path> <subset> <label>
```

Ví dụ:

```text
audio/vad_0p5_1p0/MLAAD_v7__fake_abc123__seg000.wav train spoof
audio/vad_1p0_1p5/K_SASV__utt987__seg000.wav train bonafide
audio/observed_short_0p5_2p0/Kipot__utt555__seg000.wav test spoof
```

---

## 4. Duration bins cần tạo

Vì mục tiêu là ultra-short / short speech, chốt 4 nhóm chính:

```text
vad_0p5_1p0:             speech segment từ 0.5s đến <1.0s
vad_1p0_1p5:             speech segment từ 1.0s đến <1.5s
vad_1p5_2p0:             speech segment từ 1.5s đến <=2.0s
observed_short_0p5_2p0:  toàn bộ utterance có speech_duration tự nhiên từ 0.5s đến <=2.0s
```

Không tạo:

```text
fixed_0p5
fixed_1p0
fixed_1p5
fixed_2p0
fixed_3p0
fixed_4p0
```

Nếu audio có speech dài hơn 2s, có thể cắt ra tối đa một hoặc vài speech-active segment ngắn tùy split.

---

## 5. Quy tắc chọn segment bằng VAD

### 5.1. VAD model

Dùng code tham khảo:

```text
~/code/preprocessors/models/silero_vad.py
```

Có thể customize các điểm sau:

```text
sample_rate = 16000
threshold = 0.5
min_speech_duration_ms = 250
min_silence_duration_ms = 100
speech_pad_ms = 30
window_size_samples = 512
```

Config ban đầu đề xuất:

```yaml
vad:
  model: silero_vad
  source_code_reference: ~/code/preprocessors/models/silero_vad.py
  sample_rate: 16000
  threshold: 0.5
  min_speech_duration_ms: 250
  min_silence_duration_ms: 100
  speech_pad_ms: 30
  merge_close_segments_ms: 200
```

### 5.2. Merge speech segments

Sau khi VAD trả về nhiều đoạn speech, nên merge các đoạn gần nhau nếu silence gap nhỏ:

```text
Nếu gap giữa 2 speech segments <= 200 ms:
    merge thành một speech region
```

Lý do: tránh cắt một câu nói thành nhiều mảnh quá nhỏ chỉ vì pause ngắn.

### 5.3. Loại file không đủ speech

Bỏ qua file nếu:

```text
total_speech_duration < 0.5s
```

Ghi vào `failed_files.csv` với reason:

```text
insufficient_speech_duration
```

### 5.4. Observed-short

Nếu sau VAD:

```text
0.5s <= total_speech_duration <= 2.0s
```

thì tạo một file trong:

```text
audio/observed_short_0p5_2p0/
```

Segment này nên cover từ speech_start đầu tiên đến speech_end cuối cùng, có pad nhẹ nếu cần.

### 5.5. VAD short segment từ utterance dài

Nếu:

```text
total_speech_duration > 2.0s
```

thì chọn/cắt speech-active segment theo bin:

```text
0.5–1.0s
1.0–1.5s
1.5–2.0s
```

Quy tắc chọn segment đề xuất:

1. Tạo candidate từ từng merged speech region.
2. Nếu region nằm trong một bin, dùng nguyên region.
3. Nếu region dài hơn 2s, trích đoạn con speech-active dài 1.5–2.0s.
4. Ưu tiên đoạn có năng lượng ổn định và ít silence nhất.
5. Không lấy quá nhiều segment từ cùng một utterance để tránh bias.

---

## 6. Đề xuất số lượng nên process

Bạn đang có khoảng:

```text
test:  ~1,000,000 utterances
train:   544,750 utterances
dev:     100,000 utterances
```

Chốt lại: **pre-process pool thì nên làm rộng**, nhưng **fine-tune round đầu không nên lấy 250k–300k ngay**. Với mục tiêu practical, round đầu chỉ cần đủ để kiểm tra tín hiệu VAD-short adaptation. Nếu setting/VAD threshold/sampling ratio chưa chuẩn mà train quá lớn thì rất tốn compute và khó iterate.

---

### Stage 1 — Process full một segment chính trên toàn bộ data

Đây vẫn là stage nên làm trước. Lý do: pre-process một lần để tạo pool đầy đủ, sau đó mọi experiment chỉ thay đổi protocol/subset sampling, không phải cắt audio lại.

| Split | Input size | Nên process | Segment/utt | Output dự kiến | Mục đích |
|---|---:|---:|---:|---:|---|
| train | 544,750 | toàn bộ readable files | 1 main VAD segment/utt | khoảng 450k–545k segments | tạo training pool |
| dev | 100,000 | toàn bộ readable files | 1 main VAD segment/utt | khoảng 80k–100k segments | validation pool ổn định |
| test | 1,000,000 | toàn bộ readable files | 1 main VAD segment/utt | khoảng 800k–1M segments | benchmark pool đầy đủ |

`Output dự kiến` thấp hơn input vì một số file có thể:

```text
không đọc được
không đủ speech >=0.5s
path lỗi
label lỗi
VAD fail
```

Với Stage 1, mỗi utterance chỉ sinh tối đa 1 segment chính:

```text
Nếu total_speech_duration <= 2s:
    tạo observed_short_0p5_2p0
else:
    tạo một VAD segment ưu tiên bin 1.5–2.0s
```

Lý do ưu tiên `1.5–2.0s` cho utterance dài:

```text
- đủ thông tin hơn 0.5–1.0s
- vẫn nằm trong mục tiêu short
- ít noise label hơn ultra-short cực ngắn
- phù hợp làm baseline/fine-tune ban đầu
```

---

### Stage 2 — Tạo subset nhỏ cho pilot fine-tuning

Đây là thay đổi quan trọng so với bản trước. **Không lấy 250k–300k cho round đầu.**

Đề xuất round đầu:

```text
pilot fine-tune train subset: 50k–80k segments
pilot fine-tune dev subset:   10k segments
quick eval subset:             5k/dataset
medium eval subset:           50k/dataset nếu pilot thắng
full eval:                    dùng toàn bộ processed test pool khi cần chốt
```

Mình khuyên chọn mặc định:

```text
train_pilot_80k_balanced.protocol.txt
dev_pilot_10k_balanced.protocol.txt
```

Composition cho pilot 80k:

| Nguồn segment | Tỷ lệ | Số lượng nếu 80k | Ghi chú |
|---|---:|---:|---|
| observed_short_0p5_2p0 | 20% | 16k | lấy toàn bộ nếu ít |
| vad_0p5_1p0 | 25% | 20k | kiểm tra ultra-short rõ nhất |
| vad_1p0_1p5 | 25% | 20k | vùng chuyển giữa 1s và 2s |
| vad_1p5_2p0 | 20% | 16k | short nhưng ít nhiễu hơn |
| anchor_2p0_4p0 hoặc normal_duration | 10% | 8k | chống quên MDT gốc |

Nếu chưa muốn dùng anchor 2–4s ở pre-process VAD-only, có thể thay `anchor_2p0_4p0` bằng `vad_1p5_2p0`, nhưng vẫn nên giữ một validation riêng cho 3–4s từ pipeline cũ hoặc data gốc để phát hiện catastrophic forgetting.

Nếu observed-short ít hơn 16k thì lấy toàn bộ observed-short, phần thiếu bù vào `vad_0p5_1p0`, `vad_1p0_1p5`, và `vad_1p5_2p0` theo tỷ lệ còn lại.

Label balance trong pilot train subset:

```text
bonafide: 50%
spoof:    50%
```

Nếu không thể cân bằng tuyệt đối do dataset skew, ưu tiên:

```text
1. cân bằng label trong từng duration bin nếu có thể
2. nếu không đủ sample, dùng weighted sampler khi training
3. không oversample một file gốc quá nhiều lần
```

Dataset balance:

```text
Không để một dataset chiếm >30% pilot fine-tune subset nếu có thể.
Không để một language/domain chiếm quá mạnh nếu metadata có language/domain.
```

---

### Stage 3 — Serious fine-tune nếu pilot có tín hiệu tốt

Chỉ chạy stage này nếu pilot 50k–80k cho thấy tín hiệu rõ:

```text
0.5–1s EER giảm
1–2s không giảm
FPR bonafide short không tăng mạnh
3–4s anchor không sập
```

Khi đó mới scale lên:

```text
serious fine-tune train subset: 100k–150k segments
serious fine-tune dev subset:   20k–30k segments
```

Composition đề xuất cho 150k:

| Nguồn segment | Tỷ lệ | Số lượng nếu 150k |
|---|---:|---:|
| observed_short_0p5_2p0 | 20% | 30k |
| vad_0p5_1p0 | 25% | 37.5k |
| vad_1p0_1p5 | 25% | 37.5k |
| vad_1p5_2p0 | 20% | 30k |
| anchor_2p0_4p0 hoặc normal_duration | 10% | 15k |

---

### Stage 4 — Final large fine-tune, optional

250k–300k không sai, nhưng chỉ nên dùng cho final model sau khi pilot/serious run đã chứng minh hướng này hiệu quả.

Điều kiện để chạy 200k–250k:

```text
- pilot 80k thắng baseline MDT ở ultra-short
- serious 100k–150k tiếp tục ổn định
- VAD threshold và sampling ratio đã được chốt
- không có dấu hiệu overfit theo dataset/language
- 3–4s hoặc normal-duration performance không giảm quá mức chấp nhận được
```

Quota final đề xuất:

```text
final fine-tune train subset: 200k–250k segments
final fine-tune dev subset:   30k–50k segments
```

Không nên dùng 300k ngay trừ khi compute rất thoải mái và bạn đã có hard-case mining tốt.

---

### Stage 5 — Optional multi-segment expansion

Chỉ làm nếu Stage 1 pool chưa đủ đa dạng hoặc cần tăng sample cho final run.

Tạo thêm tối đa 2–3 segments/utt cho train בלבד:

```text
max_segments_per_utt_train = 3
max_segments_per_utt_dev = 1
max_segments_per_utt_test = 1
```

Không nên multi-segment test ngay từ đầu vì sẽ làm benchmark phình to và sample correlation cao.

Nếu cần multi-segment train, quota pool mở rộng có thể là:

```text
train expansion pool target: 600k–900k segments
```

Nhưng mỗi run fine-tune vẫn nên sample theo stage:

```text
pilot:   50k–80k
serious: 100k–150k
final:   200k–250k
```

---

## 7. Quy tắc tạo segment chính cho mỗi utterance

Pseudo-rule:

```python
if total_speech_duration < 0.5:
    skip(reason="insufficient_speech_duration")

elif total_speech_duration <= 2.0:
    create_observed_short_segment()

else:
    if has_candidate_in_bin("1.5_2.0"):
        create_vad_segment(bin="1.5_2.0")
    elif has_candidate_in_bin("1.0_1.5"):
        create_vad_segment(bin="1.0_1.5")
    elif has_candidate_in_bin("0.5_1.0"):
        create_vad_segment(bin="0.5_1.0")
    else:
        create_subsegment_from_longest_speech_region(duration=2.0)
```

Với Stage 2 balanced subset, có thể tạo thêm candidate ở các bin khác từ cùng utterance, nhưng nên kiểm soát `max_segments_per_utt`.

---

## 8. File naming convention

Tên file segment cần trace được về dataset gốc nhưng không quá dài.

Format đề xuất:

```text
{dataset_name}__{orig_subset}__{label}__{utt_hash}__seg{seg_idx:03d}__{bin}.wav
```

Ví dụ:

```text
MLAAD_v7__test__spoof__a83f91c2__seg000__vad_1p5_2p0.wav
K_SASV_Bonafide__train__bonafide__e19c33aa__seg000__observed_short_0p5_2p0.wav
```

Lưu ý:

```text
- Không dùng full original path trong filename vì quá dài.
- Hash original_path để tránh trùng.
- Trace chi tiết nằm trong manifest.
```

---

## 9. Manifest schema

Mỗi split cần có:

```text
manifest.parquet
manifest.csv
protocol.txt
```

Schema đề xuất:

```text
segment_id
rel_path
abs_path
subset
label_text
label_id
source_root
source_dataset
source_protocol
source_rel_path
source_abs_path
source_subset
utt_id
utt_hash
segment_index
segment_type
duration_bin
sample_rate
num_samples
segment_duration_sec
original_duration_sec
total_speech_duration_sec
speech_ratio
vad_threshold
vad_speech_start_sec
vad_speech_end_sec
segment_start_sec
segment_end_sec
used_padding_sec
is_observed_short
is_vad_segment
vad_num_regions
vad_region_index
selection_rule
read_status
error_message
```

`protocol.txt` chỉ cần 3 cột để tương thích code hiện có:

```text
<rel_path> <subset> <label>
```

Manifest giữ toàn bộ thông tin audit.

---

## 10. Audio output format

Đề xuất:

```text
format: wav
sample_rate: 16000
channel: mono
subtype: PCM_16
```

Nếu storage quá lớn, có thể dùng FLAC:

```text
format: flac
sample_rate: 16000
channel: mono
```

Ước lượng dung lượng nếu WAV 16kHz mono PCM16:

```text
1 giây audio ≈ 32 KB
2 giây audio ≈ 64 KB
1,000,000 segments x 2s ≈ 64 GB
1,600,000 segments x 2s ≈ 102 GB
```

Vì vậy cắt file nhỏ là khả thi nếu chỉ 1 segment chính/utterance.

---

## 11. Quality control / audit

Sau khi process xong, cần sinh report cho từng split:

```text
train/audit_report.md
dev/audit_report.md
test/audit_report.md
metadata/global_audit_report.md
```

Nội dung audit report:

```text
1. Tổng số dòng protocol input
2. Tổng số file readable
3. Tổng số file failed
4. Tổng số file insufficient speech
5. Tổng số segment output
6. Distribution theo label
7. Distribution theo dataset
8. Distribution theo duration_bin
9. Mean/median/min/max segment duration
10. Mean/median total_speech_duration
11. Top failed reasons
12. Random 100 samples for manual listening
```

Cần tạo thêm file:

```text
metadata/audit_samples.csv
```

Gồm random sample để nghe thủ công:

```text
segment_id
rel_path
label
source_dataset
duration_bin
segment_duration_sec
source_abs_path
```

Quota audit thủ công đề xuất:

```text
train: 100 samples
 dev: 100 samples
test: 100 samples
```

Mỗi split nên sample đều theo:

```text
label x duration_bin x dataset
```

---

## 12. Failed files

Tất cả lỗi ghi vào:

```text
metadata/failed_files.csv
```

Schema:

```text
source_root
source_dataset
source_protocol
source_rel_path
source_abs_path
subset
label
error_type
error_message
```

Error type chuẩn:

```text
missing_path
unreadable_audio
unsupported_format
invalid_duration
invalid_label
vad_failed
insufficient_speech_duration
write_failed
unknown_error
```

Không nên để lỗi làm dừng toàn bộ job, trừ khi:

```text
failed rate > 10% trong một dataset
```

Nếu failed rate > 10%, dừng dataset đó để kiểm tra path/protocol mapping.

---

## 13. Preprocess config nên lưu

File:

```text
metadata/preprocess_config.yaml
```

Nội dung đề xuất:

```yaml
run_name: add_vad_short_29May2026
output_root: /data/add_processed_vad_short_29May2026

input_roots:
  benchmark: /data/add_eval_data
  training: /data/dsd_corpus_pool_13May2026

protocol_format: "<rel_path> <subset> <label>"

splits:
  train:
    target_input_count: 544750
    max_segments_per_utt: 1
  dev:
    target_input_count: 100000
    max_segments_per_utt: 1
  test:
    target_input_count: 1000000
    max_segments_per_utt: 1

audio:
  sample_rate: 16000
  mono: true
  output_format: wav
  wav_subtype: PCM_16

vad:
  model: silero_vad
  source_code_reference: ~/code/preprocessors/models/silero_vad.py
  threshold: 0.5
  min_speech_duration_ms: 250
  min_silence_duration_ms: 100
  speech_pad_ms: 30
  merge_close_segments_ms: 200

segment_policy:
  enable_fixed_window: false
  enable_vad_segments: true
  enable_observed_short: true
  min_total_speech_duration_sec: 0.5
  max_short_speech_duration_sec: 2.0
  main_segment_priority:
    - observed_short_0p5_2p0
    - vad_1p5_2p0
    - vad_1p0_1p5
    - vad_0p5_1p0
  duration_bins:
    vad_0p5_1p0: [0.5, 1.0]
    vad_1p0_1p5: [1.0, 1.5]
    vad_1p5_2p0: [1.5, 2.0]
    observed_short_0p5_2p0: [0.5, 2.0]

selection:
  train:
    max_segments_per_utt: 1
    prefer_bin: vad_1p5_2p0
  dev:
    max_segments_per_utt: 1
    prefer_bin: vad_1p5_2p0
  test:
    max_segments_per_utt: 1
    prefer_bin: vad_1p5_2p0

qc:
  stop_dataset_if_failed_rate_over: 0.10
  audit_samples_per_split: 100
```

---

## 14. Execution order

### Step 1 — Build raw metadata

Input:

```text
/data/add_eval_data/*/protocol.txt
/data/dsd_corpus_pool_13May2026/15_May_full.txt hoặc protocol tương ứng
```

Output:

```text
metadata/metadata_raw_all.parquet
metadata/metadata_raw_all.csv
```

Việc cần làm:

```text
- parse protocol
- resolve absolute path
- normalize label
- assign split
- assign source_dataset
- check file exists
- read audio metadata
```

---

### Step 2 — Run VAD

Input:

```text
metadata/metadata_raw_all.parquet
```

Output:

```text
metadata/vad_segments_all.parquet
metadata/failed_files.csv
```

Việc cần làm:

```text
- load audio
- resample 16k mono
- run Silero VAD
- merge close speech regions
- compute total_speech_duration
- mark observed-short candidates
- mark insufficient speech
```

---

### Step 3 — Select segment candidates

Input:

```text
metadata/vad_segments_all.parquet
```

Output:

```text
metadata/segment_candidates_all.parquet
```

Việc cần làm:

```text
- tạo observed_short nếu total_speech_duration 0.5–2.0s
- tạo VAD candidate bin 0.5–1.0s
- tạo VAD candidate bin 1.0–1.5s
- tạo VAD candidate bin 1.5–2.0s
- chọn main segment theo priority
```

---

### Step 4 — Cut audio files

Input:

```text
metadata/segment_candidates_all.parquet
```

Output:

```text
train/audio/**/*.wav
dev/audio/**/*.wav
test/audio/**/*.wav
```

Việc cần làm:

```text
- load original audio
- resample 16k mono
- cut segment_start_sec -> segment_end_sec
- write wav/flac
- verify output duration
- update manifest
```

---

### Step 5 — Write protocol and manifest

Output:

```text
train/protocol.txt
train/manifest.parquet
train/manifest.csv

dev/protocol.txt
dev/manifest.parquet
dev/manifest.csv

test/protocol.txt
test/manifest.parquet
test/manifest.csv
```

Protocol format:

```text
<rel_path> <subset> <label>
```

---

### Step 6 — QC and audit

Output:

```text
train/audit_report.md
dev/audit_report.md
test/audit_report.md
metadata/global_audit_report.md
metadata/audit_samples.csv
```

Kiểm tra bắt buộc:

```text
- output file tồn tại
- duration đúng bin
- sample rate đúng 16k
- mono
- label distribution không lệch bất thường
- dataset distribution không bị thiếu dataset
- failed rate từng dataset
```

---

## 15. Sampling policy sau khi đã process

Sau khi đã có full processed pool, tạo các file subset riêng cho experiment.

Ví dụ:

```text
subsets/
├── train_300k_balanced.protocol.txt
├── train_300k_balanced.manifest.parquet
├── dev_50k_balanced.protocol.txt
├── quick_eval_5k_per_dataset.protocol.txt
├── medium_eval_50k_per_dataset.protocol.txt
└── full_eval.protocol.txt
```

### 15.1. Pilot train 80k balanced

Đây là subset fine-tune mặc định cho round đầu.

```text
total: 80,000
bonafide: 40,000
spoof:    40,000
```

Duration/source mix:

```text
observed_short_0p5_2p0:       20%  = 16k
vad_0p5_1p0:                  25%  = 20k
vad_1p0_1p5:                  25%  = 20k
vad_1p5_2p0:                  20%  = 16k
anchor_2p0_4p0 / normal:      10%  =  8k
```

Nếu muốn sanity run nhanh hơn:

```text
train_pilot_50k_balanced.protocol.txt
dev_pilot_10k_balanced.protocol.txt
```

Nếu pilot tốt mới tạo thêm:

```text
train_serious_150k_balanced.protocol.txt
train_final_250k_balanced.protocol.txt
```

### 15.2. Dev cho pilot và scale-up

Round đầu nên dùng dev nhỏ nhưng balanced:

```text
dev_pilot_10k_balanced.protocol.txt
```

Khi scale lên serious/final run:

```text
dev_serious_20k_30k_balanced.protocol.txt
dev_final_30k_50k_balanced.protocol.txt
```

Full dev 100k vẫn nên giữ lại để kiểm tra cuối, nhưng không cần dùng cho mọi epoch nếu quá tốn thời gian.

Dev nên giữ balanced theo:

```text
label x duration_bin x dataset
```

### 15.3. Quick eval

```text
5k/dataset
```

Dùng để chọn checkpoint nhanh.

### 15.4. Medium eval

```text
50k/dataset
```

Dùng trước khi chạy full eval.

### 15.5. Full eval

```text
Toàn bộ processed test pool
```

Chỉ chạy khi checkpoint đã tốt ở quick + medium.

---

## 16. Checklist hoàn thành pre-process

Pre-process được xem là xong khi có đủ:

```text
[ ] metadata_raw_all.parquet
[ ] vad_segments_all.parquet
[ ] failed_files.csv
[ ] train/protocol.txt
[ ] train/manifest.parquet
[ ] dev/protocol.txt
[ ] dev/manifest.parquet
[ ] test/protocol.txt
[ ] test/manifest.parquet
[ ] global_audit_report.md
[ ] audit_samples.csv
[ ] train_pilot_80k_balanced.protocol.txt
[ ] dev_pilot_10k_balanced.protocol.txt
[ ] train_serious_150k_balanced.protocol.txt nếu pilot tốt
[ ] train_final_250k_balanced.protocol.txt nếu cần final model
[ ] dev_full.protocol.txt hoặc dev_final_30k_50k_balanced.protocol.txt
[ ] quick_eval_5k_per_dataset.protocol.txt
[ ] medium_eval_50k_per_dataset.protocol.txt
[ ] full_eval.protocol.txt
```

---

## 17. Khuyến nghị cuối cùng

Chốt phương án nên làm:

```text
1. Không pre-process fixed-window trong pipeline chính.
2. Dùng VAD-only + observed-short.
3. Cắt audio thành file nhỏ thật để dễ audit.
4. Stage 1 process toàn bộ train/dev/test, mỗi utterance tối đa 1 segment chính.
5. Stage 2 tạo pilot train subset 50k–80k, mặc định 80k balanced.
6. Dev pilot dùng 10k balanced; full dev 100k chỉ dùng để kiểm tra cuối hoặc khi compute ổn.
7. Nếu pilot tốt, scale lên serious 100k–150k; chỉ dùng 200k–250k cho final model.
8. Test tạo quick 5k/dataset, medium 50k/dataset, và full eval từ toàn bộ processed test.
9. Manifest phải trace được đầy đủ từ segment về original file.
```

Với dữ liệu hiện tại, con số mình đề xuất là:

```text
Full pre-process pool:
- train: process toàn bộ ~544,750 utterances
- dev:   process toàn bộ ~100,000 utterances
- test:  process toàn bộ ~1,000,000 utterances

Fine-tune thực tế:
- pilot train subset: 50k–80k segments, mặc định 80k
- pilot dev subset: 10k segments
- serious train subset nếu pilot tốt: 100k–150k segments
- final train subset nếu cần: 200k–250k segments

Evaluation:
- quick: 5k/dataset
- medium: 50k/dataset
- full: toàn bộ processed test pool
```
