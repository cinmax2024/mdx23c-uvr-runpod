import base64, os, traceback, uuid
import runpod
from audio_separator.separator import Separator

OUT_DIR = "/tmp/uvr_out"
os.makedirs(OUT_DIR, exist_ok=True)

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
        return result
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1500:],
        }


runpod.serverless.start({"handler": handler})
