FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ARG CACHE_BUST=9

# Real filesystem layer that depends on CACHE_BUST — guarantees all layers
# below get rebuilt with new content hashes when CACHE_BUST changes.
RUN echo "CACHE_BUST=${CACHE_BUST}" > /cache_bust.txt

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# FIX: Upgrade torch + torchaudio AS A MATCHED PAIR before audio-separator
# runs. audio-separator[gpu]==0.30.0 requires torch>=2.4 and was silently
# upgrading torch while leaving torchaudio at the base image's 2.2.0 — that
# ABI mismatch crashed `import torchaudio` and killed every cold start.
RUN pip install --no-cache-dir \
    torch==2.4.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Now audio-separator finds torch already at 2.4.1 and won't touch it.
RUN pip install --no-cache-dir runpod "audio-separator[gpu]==0.30.0"

RUN pip install --no-cache-dir "transformers==4.36.2" librosa soundfile

# UVR model (already proven to work — leave as-is)
RUN python -c "from audio_separator.separator import Separator; \
  s = Separator(model_file_dir='/models'); \
  s.download_model_files('MDX23C-8KFFT-InstVoc_HQ.ckpt')"

# SER model: download lazily on first request (~10-15s extra on cold start once)
WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
