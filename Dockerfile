FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ARG CACHE_BUST=8

# Real filesystem layer that depends on CACHE_BUST — guarantees all layers
# below get rebuilt with new content hashes when CACHE_BUST changes.
RUN echo "CACHE_BUST=${CACHE_BUST}" > /cache_bust.txt

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Split pip installs into two layers so each has a different hash than
# the previous failed builds. Also makes future updates faster.
RUN pip install --no-cache-dir runpod "audio-separator[gpu]==0.30.0"
RUN pip install --no-cache-dir "transformers==4.36.2" librosa soundfile

# UVR model (always pre-cache — it's already proven to work in builds)
RUN python -c "from audio_separator.separator import Separator; \
  s = Separator(model_file_dir='/models'); \
  s.download_model_files('MDX23C-8KFFT-InstVoc_HQ.ckpt')"

# SER model: download lazily on first request (~10-15s extra on cold start once)
WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
