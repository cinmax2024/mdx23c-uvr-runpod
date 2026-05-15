import base64, os, traceback, uuid, urllib.request
import runpod
from audio_separator.separator import Separator
import librosa

OUT_DIR = "/tmp/uvr_out"
os.makedirs(OUT_DIR, exist_ok=True)

# UVR (loads at module import — proven to work)
SEP = Separator(
    model_file_dir="/models",
    output_dir=OUT_DIR,
    output_format="FLAC",
)
SEP.load_model(model_filename="MDX23C-8KFFT-InstVoc_HQ.ckpt")

# SER — lazy-loaded on first use
_SER_PIPE = None
_SER_LABEL_MAP = {"ang": "ANGRY", "sad": "SAD", "hap": "HAPPY", "neu": "NEUTRAL"}


def _get_ser_pipe():
    global _SER_PIPE
    if _SER_PIPE is None:
        from transformers import pipeline as _hf_pipeline
        _SER_PIPE = _hf_pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
        )
    return _SER_PIPE


def _detect_sample_rate(path):
    try:
        import soundfile as sf
        return int(sf.info(path).samplerate)
    except Exception:
        return 44100


def _resolve(path_or_basename):
    if os.path.isabs(path_or_basename) and os.path.exists(path_or_basename):
        return path_or_basename
    candidate = os.path.join(OUT_DIR, os.path.basename(path_or_basename))
    return candidate if os.path.exists(candidate) else None


def _classify_segment_emotions(vocals_wav_path, segments):
    if not segments or not vocals_wav_path or not os.path.exists(vocals_wav_path):
        return []
    try:
        ser = _get_ser_pipe()
    except Exception as e:
        return [{"index": -1, "error": f"SER load failed: {type(e).__name__}: {e}"}]
    try:
        y, sr = librosa.load(vocals_wav_path, sr=16000, mono=True)
    except Exception:
        return []
    out = []
    for seg in segments:
        try:
            start_ms = int(seg.get("start_ms") or 0)
            end_ms = int(seg.get("end_ms") or 0)
            idx = seg.get("index")
            if idx is None or end_ms <= start_ms:
                continue
            a = max(0, int(start_ms / 1000.0 * sr))
            b = min(len(y), int(end_ms / 1000.0 * sr))
            snippet = y[a:b]
            if len(snippet) < int(sr * 0.4):
                continue
            preds = ser({"raw": snippet, "sampling_rate": sr}, top_k=4)
            top = preds[0]
            out.append({
                "index": idx,
                "emotion": _SER_LABEL_MAP.get(top["label"].lower(), "NEUTRAL"),
                "confidence": float(top["score"]),
                "scores": {p["label"]: float(p["score"]) for p in preds[:4]},
            })
        except Exception:
            continue
    return out


def _download_url(url, dest_path, timeout=120):
    """Download a URL to dest_path. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RunPodHandler/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        return os.path.getsize(dest_path) > 1024
    except Exception as e:
        print(f"[handler] download failed: {type(e).__name__}: {e}")
        return False


def _upload_to_presigned(file_path, put_url, timeout=120):
    """PUT a local file to a presigned upload URL. Returns True on success."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        req = urllib.request.Request(
            put_url,
            data=data,
            method="PUT",
            headers={"Content-Type": "audio/flac"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception as e:
        print(f"[handler] upload failed: {type(e).__name__}: {e}")
        return False


def handler(event):
    try:
        inp = event.get("input") or {}
        audio_b64 = inp.get("audio_b64")
        audio_url = inp.get("audio_url")
        if not audio_b64 and not audio_url:
            return {"ok": False, "error": "missing input.audio_b64 or input.audio_url"}
        fmt = (inp.get("audio_format") or "flac").lower()
        seg_input = inp.get("segments")
        instr_upload_url = inp.get("instrumental_upload_url")
        vocals_upload_url = inp.get("vocals_upload_url")
        url_mode = bool(audio_url)

        rid = uuid.uuid4().hex[:10]
        in_path = os.path.join(OUT_DIR, f"req_{rid}.{fmt}")

        if url_mode:
            if not _download_url(audio_url, in_path):
                return {"ok": False, "error": "failed to download audio_url"}
        else:
            with open(in_path, "wb") as f:
                f.write(base64.b64decode(audio_b64))

        stems = SEP.separate(in_path)

        instr_path = None
        vocals_path = None
        for s in stems:
            bn = os.path.basename(s).lower()
            p = _resolve(s)
            if not p:
                continue
            if "instrument" in bn or "_inst" in bn:
                instr_path = p
            elif "vocal" in bn:
                vocals_path = p

        if not instr_path:
            return {
                "ok": False,
                "error": "instrumental file not found on disk",
                "stems_returned": stems,
                "out_dir_listing": sorted(os.listdir(OUT_DIR))[:30],
            }

        sr = _detect_sample_rate(instr_path)

        # SER on the clean vocals stem
        emotions = []
        if seg_input and vocals_path:
            emotions = _classify_segment_emotions(vocals_path, seg_input)

        result = {
            "ok": True,
            "format": "flac",
            "sample_rate": sr,
        }

        if url_mode:
            # Upload stems to presigned URLs the worker provided
            if instr_upload_url:
                ok = _upload_to_presigned(instr_path, instr_upload_url)
                result["instrumental_uploaded"] = ok
                if not ok:
                    result["error"] = "instrumental upload to presigned URL failed"
            if vocals_path and vocals_upload_url:
                ok = _upload_to_presigned(vocals_path, vocals_upload_url)
                result["vocals_uploaded"] = ok
        else:
            # Base64 mode (small audio path) — keep existing behaviour
            with open(instr_path, "rb") as f:
                result["instrumental_b64"] = base64.b64encode(f.read()).decode("ascii")
            if vocals_path:
                with open(vocals_path, "rb") as f:
                    result["vocals_b64"] = base64.b64encode(f.read()).decode("ascii")

        if emotions:
            result["emotions"] = emotions

        # Cleanup local
        for p in [in_path, instr_path, vocals_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        return result
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1500:],
        }


runpod.serverless.start({"handler": handler})
