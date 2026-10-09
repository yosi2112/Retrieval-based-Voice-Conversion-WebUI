"""Check the Colab libraries and RVC imports without starting training."""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-models", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    os.environ["RVC_CUDA_GRAPH"] = "0"
    import torch
    import torchaudio
    import train

    if Path(train.__file__).resolve() != PROJECT_ROOT / "train" / "__init__.py":
        raise RuntimeError(f"Wrong training package: {train.__file__}")
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "torchaudio", "numpy", "scipy", "librosa", "gradio", "transformers",
    )}
    versions.update({"python": sys.version, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
    print(json.dumps(versions, indent=2), flush=True)
    if torch.__version__.split("+")[0] != torchaudio.__version__.split("+")[0]:
        raise RuntimeError("torch and torchaudio releases differ.")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before launching the WebUI.")
    missing_commands = [command for command in ("ffmpeg", "ffprobe") if not shutil.which(command)]
    if missing_commands:
        raise RuntimeError(f"Missing system commands: {missing_commands}")
    for module in (
        "train.utils", "train.data_utils", "train.losses", "train.mel_processing", "train.process_ckpt",
        "configs.config", "infer.module.models", "infer.hubert", "infer.rmvpe", "infer.fcpe", "infer.vc.modules",
        "tools.gradio_compat",
    ):
        importlib.import_module(module)
        print(f"Import OK: {module}", flush=True)
    import gradio as gr
    from tools.gradio_compat import audio_upload_options, blocks_css_options, queue_app
    with gr.Blocks(**blocks_css_options("")) as app:
        audio = gr.Audio(type="filepath", **audio_upload_options())
        result = gr.Textbox()
        button = gr.Button("Check")
        button.click(lambda path: path or "", [audio], [result])
    queue_app(app)
    from train.mel_processing import mel_spectrogram_torch
    device = "cuda" if args.require_cuda else "cpu"
    signal = torch.randn(1, 8000, device=device, requires_grad=True)
    mel = mel_spectrogram_torch(signal, 1024, 80, 40000, 400, 1024, 0, 20000)
    mel.mean().backward()
    if not torch.isfinite(mel).all() or not torch.isfinite(signal.grad).all():
        raise RuntimeError("Spectrogram forward/backward produced non-finite values.")
    if args.require_models:
        required = [PROJECT_ROOT / "assets/hubert_base/config.json", PROJECT_ROOT / "assets/rmvpe/rmvpe.pt"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Missing model files: {missing}")
        from infer.hubert import load_hubert_model
        load_hubert_model(torch.device(device))
    print("RVC compatibility checks passed.", flush=True)


if __name__ == "__main__":
    main()
