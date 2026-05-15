FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod "audio-separator[gpu]==0.30.0"

# Pre-download MDX23C weights at build time so first request is fast
RUN python -c "from audio_separator.separator import Separator; \
  s = Separator(model_file_dir='/models'); \
  s.download_model_files('MDX23C-8KFFT-InstVoc_HQ.ckpt')"

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
