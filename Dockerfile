# -------------------------------------------------------------------
# Base image: PyTorch with CUDA (matches F5-TTS requirements)
# -------------------------------------------------------------------
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# -------------------------------------------------------------------
# System dependencies (REQUIRED)
# -------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    sox \
    libsndfile1 \
    libgl1 \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Set working directory
# -------------------------------------------------------------------
WORKDIR /workspace

# -------------------------------------------------------------------
# Install Python dependencies FIRST (faster rebuilds)
# -------------------------------------------------------------------
COPY requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

# -------------------------------------------------------------------
# Clone and install F5-TTS (MANDATORY)
# -------------------------------------------------------------------
RUN git clone https://github.com/SWivid/F5-TTS.git && \
    cd F5-TTS && \
    git submodule update --init --recursive && \
    pip install -e .

# -------------------------------------------------------------------
# Set working directory to F5-TTS root
# -------------------------------------------------------------------
WORKDIR /workspace/F5-TTS

# -------------------------------------------------------------------
# Copy your FastAPI app
# -------------------------------------------------------------------
COPY main.py /workspace/F5-TTS/main.py

# -------------------------------------------------------------------
# Expose port
# -------------------------------------------------------------------
EXPOSE 8000

# -------------------------------------------------------------------
# Start server (GPU-safe: ONE worker only)
# -------------------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
