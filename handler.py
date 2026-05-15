import base64, os, traceback, uuid
import runpod
from audio_separator.separator import Separator
from transformers import pipeline as _hf_pipeline
import librosa

OUT_DIR = "/tmp/uvr_out"
os.makedirs(OUT_DIR, exist_ok=True)

# UVR (vocal separation)
SEP = Separator(
    model_file_dir="/models",
    output_dir=OUT_DIR,
    output_format="FLAC",
)
SEP.load_model(model_filename="MDX23C-8KFFT-InstVoc_HQ.ckpt")

# SER (speech emotion recognition)
SER_PIPE = _hf_pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    top_k=4,
)
_SER_LABEL_MAP = {"ang": "ANGRY", "sad": "SAD", "hap": "HAPPY", "neu": "NEUTRAL"}


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
    """For each segment, slice the clean vocals and run SER. Returns list of
    {index, emotion, confidence, scores}. Skips segments shorter than 400ms."""
    if not segments or not vocals_wav_path or not os.path.exists(vocals_wav_path):
        return []
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
            preds = SER_PIPE({"raw": snippet, "sampling_rate": sr})
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


def handler(event):
    try:
        inp = event.get("input") or {}
        audio_b64 = inp.get("audio_b64")
        if not audio_b64:
            return {"ok": False, "error": "missing input.audio_b64"}
        fmt = (inp.get("audio_format") or "flac").lower()
        seg_input = inp.get("segments")  # optional list of {index, start_ms, end_ms}

        rid = uuid.uuid4().hex[:10]
        in_path = os.path.join(OUT_DIR, f"req_{rid}.{fmt}")
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

        with open(instr_path, "rb") as f:
            instr_bytes = f.read()
        sr = _detect_sample_rate(instr_path)

        vocals_b64 = None
        if vocals_path:
            with open(vocals_path, "rb") as f:
                vocals_b64 = base64.b64encode(f.read()).decode("ascii")

        # SER on the clean vocals stem
        emotions = []
        if seg_input and vocals_path:
            emotions = _classify_segment_emotions(vocals_path, seg_input)

        # Cleanup
        for p in [in_path, instr_path, vocals_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        result = {
            "ok": True,
            "instrumental_b64": base64.b64encode(instr_bytes).decode("ascii"),
            "format": "flac",
            "sample_rate": sr,
        }
        if vocals_b64:
            result["vocals_b64"] = vocals_b64
        if emotions:
            result["emotions"] = emotions
        return result
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1500:],
        }


runpod.serverless.start({"handler": handler})
