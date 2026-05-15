# rebuild trigger 2
import base64, os, tempfile, traceback
import runpod
from audio_separator.separator import Separator

# Load model once at cold-start (~10-20s, then reused for every request)
SEP = Separator(model_file_dir="/models", output_format="FLAC")
SEP.load_model(model_filename="MDX23C-8KFFT-InstVoc_HQ.ckpt")

def _detect_sample_rate(path):
    try:
        import soundfile as sf
        return int(sf.info(path).samplerate)
    except Exception:
        return 44100

def handler(event):
    try:
        inp = event.get("input") or {}
        audio_b64 = inp.get("audio_b64")
        if not audio_b64:
            return {"ok": False, "error": "missing input.audio_b64"}
        fmt = (inp.get("audio_format") or "flac").lower()

        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, f"input.{fmt}")
            with open(in_path, "wb") as f:
                f.write(base64.b64decode(audio_b64))

            # Tell separator to write stems INTO our temp dir
            SEP.output_dir = td
            stems = SEP.separate(in_path)

            # audio-separator returns basenames; resolve against output_dir
            stem_paths = [s if os.path.isabs(s) else os.path.join(td, s) for s in stems]
            instr = next(
                (p for p in stem_paths
                 if "instrument" in os.path.basename(p).lower()
                 or "_inst" in os.path.basename(p).lower()),
                None,
            )
            if not instr or not os.path.exists(instr):
                return {"ok": False, "error": f"no instrumental stem found, got: {stems}"}

            with open(instr, "rb") as f:
                data = f.read()
            sr = _detect_sample_rate(instr)

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
            "trace": traceback.format_exc()[-1000:],
        }

runpod.serverless.start({"handler": handler})
