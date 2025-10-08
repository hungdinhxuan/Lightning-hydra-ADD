# How to use run experiment using docker

reference docs: https://www.runpod.io/articles/guides/docker-setup-pytorch-cuda-12-8-python-3-11

## Prerequisite
- Install NVIDIA Drivers: Ensure the latest NVIDIA GPU drivers are installed on the host system.
- Install Docker: Install Docker Desktop (on Windows/macOS with WSL2 for Windows) or Docker Engine (on Linux).
- Install NVIDIA Container Toolkit: Follow the instructions for installing the NVIDIA Container Toolkit for your specific operating system and Docker installation. This typically involves adding the NVIDIA package repositories and installing the nvidia-container-toolkit package.
- Run a container with GPU support:
Using docker run: Include the --gpus flag when running a container. 

For example, to provide access to all GPUs:
```bash
docker run --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```
In case only one gpu:
```bash
docker run --gpus '"device=0"' nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```
## Build image
```bash
docker build -t <name_image> .
```

## run container
```bash
docker run --rm --name lha --gpus '"device=3"' \
  -v /nvme2:/nvme2 \
  -v /data:/data \
  -v /nvme1:/nvme1 \
  -v /nvme1/hungdx/Lightning-hydra/data:/project/data \
  -v /nvme1/hungdx/Lightning-hydra/pretrained:/project/pretrained \
  --env-file .env \
  -e OMP_NUM_THREADS=5 \
  --ipc=host \
  lightning-hydra-add
```
### lightweight-version
```bash
docker run --rm --name lhalw --gpus '"device=3"' \
  -v /nvme2:/nvme2 \
  -v /data:/data \
  -v /nvme1:/nvme1 \
  -v /nvme1/hungdx/Lightning-hydra/data:/project/data \
  -v /nvme1/hungdx/Lightning-hydra/pretrained:/project/pretrained \
  --env-file .env \
  -e OMP_NUM_THREADS=5 \
  --ipc=host \
  lightning-hydra-add-lw
```

## Test to make sure gpu avaiable
```bash
docker exec -it <container_id> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```    
## Training experiment

```bash
docker exec -it lha python src/train.py experiment=xlsr_conformertcm_mdt ++model_averaging=True
```

# References
https://mveg.es/posts/optimizing-pytorch-docker-images-cut-size-by-60percent/