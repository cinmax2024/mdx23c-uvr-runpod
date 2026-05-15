import base64, os, traceback, uuid
import runpod
from audio_separator.separator import Separator

OUT_DIR = "/tmp/uvr_out"
os.makedirs(OUT_DIR, exist_ok=True)

# Construct WITH output_dir BEFORE load_model so model_instance freezes the right path
SEP = Separator(
    model_file_dir="/models",
    output_dir=OUT_DIR,
    output_format="FLAC",
)
SEP.load_model(model_filename="MDX23C-8KFFT-InstVoc_HQ.ckpt")


def _detect_sample_rate(path):
    try:
        import soundfile as sf
        return int(sf.info(path).samplerate)
    except Exception:
        return 44100


def _resolve(path_or_basename):
    """Stems may be absolute paths OR basenames depending on audio-separator version."""
    if os.path.isabs(path_or_basename) and os.path.exists(path_or_basename):
        return path_or_basename
    candidate = os.path.join(OUT_DIR, os.path.basename(path_or_basename))
    return candidate if os.path.exists(candidate) else None


def handler(event):
    try:
        inp = event.get("input") or {}
        audio_b64 = inp.get("audio_b64")
        if not audio_b64:
            return {"ok": False, "error": "missing input.audio_b64"}
        fmt = (inp.get("audio_format") or "flac").lower()

        rid = uuid.uuid4().hex[:10]
        in_path = os.path.join(OUT_DIR, f"req_{rid}.{fmt}")
        with open(in_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))

        stems = SEP.separate(in_path)

        instr_path = None
        for s in stems:
            bn = os.path.basename(s).lower()
            if "instrument" in bn or "_inst" in bn:
                instr_path = _resolve(s)
                if instr_path:
                    break

        if not instr_path:
            # Self-diagnostic dump so we never need another build to debug paths
            return {
                "ok": False,
                "error": "instrumental file not found on disk",
                "stems_returned": stems,
                "out_dir_listing": sorted(os.listdir(OUT_DIR))[:30],
                "cwd_listing": sorted(os.listdir(os.getcwd()))[:30],
                "out_dir": OUT_DIR,
            }

        with open(instr_path, "rb") as f:
            data = f.read()
        sr = _detect_sample_rate(instr_path)

        # Cleanup all files for this request
        for p in [in_path] + [_resolve(s) for s in stems]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        return {
            "ok": True,
            "instrumental_b64": base64.b64encode(data).decode("ascii"),
            "format": "flac",
            "sample_rate": sr,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1500:],
        }


runpod.serverless.start({"handler": handler})
