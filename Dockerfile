# FASGA Dockerfile with NVIDIA CUDA and cuDNN support
# Base image: NVIDIA CUDA 12.4.1 with cuDNN 9 on Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Metadata
LABEL maintainer="FASGA Contributors"
LABEL description="Force-Aligned Subtitle Generator for Audiobooks with GPU support"
LABEL version="0.1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python and build tools
    python3.11 \
    python3.11-dev \
    python3-pip \
    # FFmpeg for audio processing
    ffmpeg \
    # Audio libraries
    libsndfile1 \
    libsndfile1-dev \
    # Other utilities
    curl \
    wget \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy uv from official image (recommended approach per https://docs.astral.sh/uv/guides/integration/docker/)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY README.md AGENTS.md SPECS.md ./
COPY check_cuda.py ./

# Install Python dependencies using uv with cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Create directory for input/output files
RUN mkdir -p /data/input /data/output

# Set volume mount points
VOLUME ["/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

# Default command: show help
CMD ["uv", "run", "fasga", "--help"]

# Usage examples (in comments):
#
# Build:
#   docker build -t fasga:latest .
#
# Run diagnostic:
#   docker run --rm --gpus all fasga:latest uv run python check_cuda.py
#
# Process audiobook:
#   docker run --rm --gpus all \
#     -v /path/to/your/files:/data \
#     fasga:latest \
#     uv run fasga /data/audio.mp3 /data/text.txt -o /data/output.srt
#
# Interactive shell:
#   docker run --rm -it --gpus all \
#     -v /path/to/your/files:/data \
#     fasga:latest bash

