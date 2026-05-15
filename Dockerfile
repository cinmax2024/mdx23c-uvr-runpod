FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ARG CACHE_BUST=5
ENV CACHE_BUST=${CACHE_BUST}

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod "audio-separator[gpu]==0.30.0" transformers librosa

# UVR model (always pre-cache — it's already proven to work in builds)
RUN python -c "from audio_separator.separator import Separator; \
  s = Separator(model_file_dir='/models'); \
  s.download_model_files('MDX23C-8KFFT-InstVoc_HQ.ckpt')"

# SER model: download lazily on first request (~10-15s extra on cold start once)
# Skipped from build to avoid the recurring HF download/import failures here.

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
