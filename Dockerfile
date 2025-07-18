# CUDA-compatible PyTorch base
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies


# Avoid tzdata prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y tzdata ffmpeg git && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*
    
# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
WORKDIR /app
COPY . .

# Run main script	docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
CMD ["python", "main.py"]