Kế hoạch warm-up trước khi training thật
Mục tiêu
Thiết lập một flow bắt buộc theo thứ tự: preflight -> convert/cache prep -> warm-up benchmark -> ghi nhận cấu hình tối ưu vào `logs/optimized_configs` -> training thật. Cách làm này giúp tránh tình huống chạy train full rồi mới phát hiện bottleneck nằm ở storage, WebDataset sharding, DataLoader, hoặc decode audio.
Training command gốc
```bash
python src/train.py \
  experiment=xlsr_conformertcm_mdt_optimized \
  ++model_averaging=True \
  ++data.data_dir=data/HABLA_novel_sets \
  ++data.args.protocol_path=data/HABLA_novel_sets/protocol.txt \
  logger=json \
  ++trainer.max_epochs=1 \
  ++json_logging.enabled=true \
  ++json_logging.log_path=/tmp/metrics.json
```
Nguyên tắc vận hành
Không chạy training thật ngay từ đầu.
Nếu đang dùng chế độ optimized/WebDataset thì phải kiểm tra dataset đã được convert sang WDS hợp lệ hay chưa.
Nếu chưa có WDS hoặc WDS stale so với dữ liệu nguồn thì phải convert trước.
Chỉ training thật sau khi warm-up benchmark hoàn tất và đã ghi markdown cấu hình được chọn vào thư mục `logs/optimized_configs`.
Mọi bước phải fail-fast nếu tài nguyên máy không đủ an toàn.
Pha 0: Preflight check
Mục tiêu
Kiểm tra trạng thái môi trường trước khi đụng vào training hoặc convert.
Điều kiện cần kiểm tra
Xác định experiment hiện tại có đang bật optimized mode/WebDataset hay không.
Kiểm tra `data.data_dir` và `protocol_path` có tồn tại.
Kiểm tra thư mục output cho WDS, cache local, `/dev/shm`, và `logs/optimized_configs`.
Kiểm tra free disk, free RAM, free `/dev/shm`.
Nếu disk hoặc RAM còn dưới 10% thì dừng và báo warning.
Kiểm tra GPU visibility và số worker dự kiến có hợp lý với server hiện tại hay không.
Output
`preflight_report.json`
`preflight_report.md`
Pha 1: Detect và convert sang WebDataset
Logic quyết định
Nếu `optimized_mode = false` -> bỏ qua convert, đi tiếp sang warm-up.
Nếu `optimized_mode = true` và chưa có WDS -> chạy convert.
Nếu `optimized_mode = true` và WDS có rồi nhưng manifest/source timestamp không khớp -> cảnh báo hoặc convert lại.
Sau convert phải verify lại shard count, sample count, split, và shard size thực tế.
Script liên quan
`scripts/convert_to_wds.py`
`scripts/convert_to_wds.md`
Chính sách shard
Shard size mục tiêu ban đầu: 512 MB - 1 GB.
Không cố định theo số sample nếu duration audio lệch nhiều.
Ưu tiên shard theo target bytes.
Output
`convert_report.json`
`convert_report.md`
`manifest.json`
Pha 2: Warm-up benchmark data-only
Mục tiêu
Chạy benchmark chỉ với DataLoader/WebDataset, chưa forward model, để đo raw input throughput.
Các biến cần sweep
`num_workers`: 2, 4, 8, 12
`prefetch_factor`: 2, 4, 8
`persistent_workers`: true/false
`pin_memory`: true/false
`cache_mode`: `none` / `nvme_cache` / `shm_hot_cache`
`hot_shard_window`: 2 / 4 / 8 shard
`shard_size_profile`: 256 MB / 512 MB / 1 GB nếu cần benchmark nhiều biến thể
Metrics cần log
`time_to_first_batch`
`steady_state_batches_per_sec`
`samples_per_sec`
CPU utilization
RAM usage
`/dev/shm` usage
local cache usage
read/decode error count
Output
`warmup_data.csv`
`warmup_data.json`
`warmup_data.md`
Pha 3: Warm-up benchmark micro-train
Mục tiêu
Chạy model thật trong 200-500 steps để xác nhận throughput cuối cùng và xem GPU có bị starvation do input pipeline hay không.
Cần đo thêm
`global_step_per_sec`
GPU utilization trung bình
data wait ratio
batch latency
loss có chạy ổn định không
có xuất hiện bottleneck ở decode hoặc host-to-device transfer hay không
Output
`warmup_train.csv`
`warmup_train.json`
`warmup_train.md`
Pha 4: Chọn cấu hình tối ưu và ghi vào docs
Quy tắc chọn cấu hình
Ưu tiên cấu hình thỏa đồng thời các điều kiện sau:
Throughput cao nhất hoặc gần cao nhất.
GPU utilization đủ cao.
RAM usage và `/dev/shm` usage không chạm ngưỡng nguy hiểm.
Không có lỗi read/decode.
`time_to_first_batch` không quá lớn.
Kết quả benchmark lặp lại đủ ổn định giữa các lần chạy.
Rule mặc định đề xuất
Reject nếu free disk < 10%.
Reject nếu free RAM < 10%.
Reject nếu `/dev/shm` pressure quá cao hoặc gây risk OOM.
Reject nếu GPU util thấp kéo dài do input bottleneck.
Approve nếu throughput tốt và tài nguyên vẫn còn headroom an toàn.
Nơi ghi nhận kết quả
Tất cả markdown tổng hợp cấu hình tối ưu phải được ghi vào:
```text
logs/optimized_configs/
```
Nội dung tối thiểu của file markdown
Machine profile: CPU, RAM, GPU, storage, `/dev/shm`
Dataset profile: source path, protocol, số sample, split, WDS path
Benchmark matrix: các config đã thử
Config được chọn: worker, prefetch, pin_memory, persistent_workers, cache mode, hot shard window
Lý do chọn
Command train thật đã được render hoàn chỉnh
Pha 5: Training thật
Chỉ được chạy khi
Preflight pass
Convert pass hoặc được xác nhận không cần convert
Warm-up data-only pass
Warm-up micro-train pass
Markdown config đã được ghi vào `logs/optimized_configs`
Gợi ý config mặc định ban đầu
Shard size: 512 MB - 1 GB
Gốc dataset trên SSD/NVMe nếu có; nếu không thì cache local trước
`cache_dir` đặt trên NVMe, không đặt toàn bộ cache ở `/dev/shm`
`/dev/shm` chỉ dùng cho 2-8 shard nóng và toàn bộ dev/eval nếu đủ RAM
`pin_memory=True`
`persistent_workers=True`
Sau đó sweep `num_workers` và `prefetch_factor`
Cấu trúc script đề xuất
1. `scripts/preflight_and_prepare.py`
Nhiệm vụ:
Detect optimized mode
Kiểm tra tài nguyên
Kiểm tra dataset path và protocol
Kiểm tra WDS đã tồn tại chưa
Quyết định có convert hay không
Tạo report preflight
2. `scripts/warmup_benchmark.py`
Nhiệm vụ:
Chạy data-only benchmark
Chạy micro-train benchmark
Sweep nhiều cấu hình DataLoader/cache
Ghi CSV/JSON/Markdown kết quả
3. `scripts/select_optimized_config.py`
Nhiệm vụ:
Đọc toàn bộ kết quả benchmark
Rank cấu hình theo rule đã định
Xuất `chosen_config.json`
Ghi markdown tổng hợp vào `logs/optimized_configs`
4. `scripts/run_training_gate.py`
Nhiệm vụ:
Chỉ cho phép training thật nếu `chosen_config.json` tồn tại và hợp lệ
Nếu chưa có thì tự động gọi preflight + warm-up + select config trước
Sau đó mới gọi `src/train.py`
Flow thực thi đề xuất
```text
run_training_gate.py
  -> preflight_and_prepare.py
  -> convert_to_wds.py (nếu cần)
  -> warmup_benchmark.py
  -> select_optimized_config.py
  -> ghi markdown vào logs/optimized_configs
  -> train thật
```
Mẫu command cho flow mới
Bước 1: Preflight
```bash
python scripts/preflight_and_prepare.py \
  experiment=xlsr_conformertcm_mdt_optimized \
  data_dir=data/HABLA_novel_sets \
  protocol_path=data/HABLA_novel_sets/protocol.txt
```
Bước 2: Warm-up
```bash
python scripts/warmup_benchmark.py \
  experiment=xlsr_conformertcm_mdt_optimized \
  data_dir=data/HABLA_novel_sets \
  protocol_path=data/HABLA_novel_sets/protocol.txt \
  --data-only-first true \
  --micro-train-steps 300 \
  --output_dir logs/optimized_configs
```
Bước 3: Chọn config
```bash
python scripts/select_optimized_config.py \
  --input_dir logs/optimized_configs \
  --output_dir logs/optimized_configs
```
Bước 4: Training thật
```bash
python scripts/run_training_gate.py \
  experiment=xlsr_conformertcm_mdt_optimized \
  ++model_averaging=True \
  ++data.data_dir=data/HABLA_novel_sets \
  ++data.args.protocol_path=data/HABLA_novel_sets/protocol.txt \
  logger=json \
  ++trainer.max_epochs=1 \
  ++json_logging.enabled=true \
  ++json_logging.log_path=/tmp/metrics.json \
  --optimized-config-dir logs/optimized_configs
```
Quy tắc fail-fast nên có
Thiếu dataset path -> fail ngay.
Thiếu protocol -> fail ngay.
Thiếu WDS trong optimized mode -> trigger convert hoặc fail có hướng dẫn.
Disk/RAM dưới ngưỡng -> fail ngay.
`/dev/shm` không đủ cho hot-cache -> fallback về NVMe cache thay vì cố dùng RAM.
Warm-up không đạt ngưỡng tối thiểu -> không cho chạy training thật.
Gợi ý tiêu chí pass tối thiểu
`time_to_first_batch` nằm trong ngưỡng chấp nhận được.
Throughput sau warm-up ổn định.
GPU utilization không thấp kéo dài do data starvation.
Không có lỗi read/decode nghiêm trọng.
Tài nguyên hệ thống còn đủ headroom sau benchmark.
Ghi chú cuối
Mục tiêu của flow này không chỉ là tăng tốc train, mà còn tạo ra một quy trình có thể lặp lại theo từng server. Mỗi máy có thể chạy warm-up riêng, tự ghi kết quả vào `logs/optimized_configs`, rồi mới chọn cấu hình tốt nhất để train thật.