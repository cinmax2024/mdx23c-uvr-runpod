FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Cache buster — bump this to force a clean rebuild
ARG CACHE_BUST=3
ENV CACHE_BUST=${CACHE_BUST}

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod "audio-separator[gpu]==0.30.0" transformers librosa

# Pre-download MDX23C UVR weights
RUN python -c "from audio_separator.separator import Separator; \
  s = Separator(model_file_dir='/models'); \
  s.download_model_files('MDX23C-8KFFT-InstVoc_HQ.ckpt')"

# Pre-download SER model so first request doesn't pay the download tax
RUN python -c "from transformers import pipeline; \
  pipeline('audio-classification', model='superb/wav2vec2-base-superb-er', top_k=4)"

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
