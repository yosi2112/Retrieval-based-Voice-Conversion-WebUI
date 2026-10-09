"""Behavior checks against the actual Python 3.13 / current ML and UI libraries."""

import ast
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["RVC_CUDA_GRAPH"] = "0"


class ColabCompatibilityTests(unittest.TestCase):
    def setUp(self):
        output = ROOT / ".test-outputs"
        output.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=output)
        self.folder = Path(self.temp.name).resolve()
        # Verify recursive cleanup stays inside our own test output directory.
        assert self.folder.is_relative_to(output.resolve())
        self.addCleanup(self.temp.cleanup)

    def test_reject_old_runtime_and_mismatched_torch_family(self):
        from tools.colab_setup import validate_runtime
        with patch("tools.colab_setup.platform.system", return_value="Linux"), patch(
            "tools.colab_setup.sys.version_info", (3, 13, 15)
        ):
            validate_runtime({"torch": "2.11.0+cu128", "torchaudio": "2.11.0+cu128"})
            with self.assertRaisesRegex(RuntimeError, "2.11"):
                validate_runtime({"torch": "2.7.1+cu118", "torchaudio": "2.7.1+cu118"})
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                validate_runtime({"torch": "2.11.0+cu128", "torchaudio": "2.12.0+cu128"})

    def test_training_dependency_imports_and_amp_gradient_step(self):
        for module in ("train.utils", "train.data_utils", "train.losses", "train.mel_processing", "train.process_ckpt"):
            importlib.import_module(module)
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with torch.amp.autocast("cuda", enabled=False):
            loss = parameter.square()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        self.assertAlmostEqual(parameter.item(), 1.6, places=5)

    def test_spectrogram_and_checkpoint_optimizer_roundtrip(self):
        from train.mel_processing import mel_spectrogram_torch
        from train.utils import save_checkpoint, load_checkpoint
        signal = torch.randn(1, 8000, requires_grad=True)
        mel = mel_spectrogram_torch(signal, 1024, 80, 40000, 400, 1024, 0, 20000)
        self.assertEqual(mel.shape[1], 80)
        mel.mean().backward()
        self.assertTrue(torch.isfinite(signal.grad).all())
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.randn(1, 3)).sum().backward()
        optimizer.step()
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        checkpoint = self.folder / "G_1.pth"
        save_checkpoint(model, optimizer, 0.01, 3, checkpoint)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        _, _, rate, epoch = load_checkpoint(checkpoint, model, optimizer)
        self.assertEqual((rate, epoch), (0.01, 3))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key])

    def test_hubert_v1_v2_local_model_loading(self):
        from transformers import HubertConfig, Wav2Vec2FeatureExtractor
        import infer.hubert as hubert
        config = HubertConfig(
            hidden_size=16, num_hidden_layers=12, num_attention_heads=2,
            intermediate_size=32, classifier_proj_size=8,
            conv_dim=(16, 16), conv_stride=(2, 2), conv_kernel=(3, 3),
            num_conv_pos_embedding_groups=2, num_conv_pos_embeddings=16,
            mask_time_prob=0.0,
        )
        model = hubert.HubertModelWithFinalProj(config).eval()
        model.save_pretrained(self.folder)
        Wav2Vec2FeatureExtractor(do_normalize=True).save_pretrained(self.folder)
        with patch.object(hubert, "HUBERT_MODEL_PATH", self.folder):
            loaded = hubert.load_hubert_model(torch.device("cpu"))
            hubert.hubert_audio_requires_normalization.cache_clear()
            self.assertTrue(hubert.hubert_audio_requires_normalization())
            source = torch.randn(1, 320)
            with torch.no_grad():
                v1 = hubert.extract_hubert_features(loaded, source, "v1")
                v2 = hubert.extract_hubert_features(loaded, source, "v2")
            self.assertEqual(v1.shape[-1], 8)
            self.assertEqual(v2.shape[-1], 16)
            self.assertTrue(torch.isfinite(v1).all() and torch.isfinite(v2).all())
        hubert.hubert_audio_requires_normalization.cache_clear()

    def test_decoder_does_not_require_torchcodec(self):
        import infer.audio as audio
        samples = np.arange(20, dtype=np.float32).reshape(2, 10)
        # A missing TorchCodec must not force CUDA resampling onto CPU.
        with patch.object(audio, "_decode_audio_ffmpeg", return_value=(samples, 48000)), patch.object(
            audio, "_resample_tensor_gpu", return_value="gpu-result"
        ) as resample:
            result = audio._load_audio_torchaudio_gpu("input.wav", 40000, False, True)
        self.assertEqual(result, "gpu-result")
        self.assertIs(resample.call_args.args[0], samples)
        self.assertEqual(resample.call_args.args[1:], (48000, 40000, False, True))

    def test_complete_webui_build_and_updates(self):
        # Run the full UI in a disposable working directory, with real Gradio
        # components/events. Launch is intercepted so no public server opens.
        shutil.copytree(ROOT / "i18n/locale", self.folder / "i18n/locale")
        source = ast.parse((ROOT / "webui.py").read_text(encoding="utf-8"))
        previous_cwd = Path.cwd()
        self.addCleanup(os.chdir, previous_cwd)
        os.chdir(self.folder)
        namespace = {"__name__": "rvc_ui_smoke", "__file__": str(ROOT / "webui.py")}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ))
            stack.enter_context(patch.object(sys, "argv", ["webui.py", "--noautoopen"]))
            launch = stack.enter_context(patch.object(gr.Blocks, "launch", return_value=None))
            exec(compile(source, str(ROOT / "webui.py"), "exec"), namespace)
        self.assertGreater(len(namespace["app"].config["components"]), 100)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launch.call_args.kwargs["css"], namespace["TRAINING_INFO_CSS"])
        update = namespace["sync_exp_name"]("new", "old")
        self.assertEqual(update["value"], "new")
        self.assertEqual(update["__type__"], "update")
        slider, dropdown = importlib.import_module("infer.vc.modules").speaker_selector_updates(
            {"speaker_info": [{"id": 0, "name": "Voice"}]}, 1
        )
        self.assertFalse(slider["visible"])
        self.assertTrue(dropdown["visible"])

    def test_notebook_is_valid_and_uses_same_python_for_workers(self):
        notebook = json.loads((ROOT / "RVC_Colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<notebook-cell>", "exec")
        self.assertIn('"--pycmd", PYTHON', "".join(notebook["cells"][-1]["source"]))

    def test_index_worker_builds_searchable_index(self):
        import faiss
        shutil.copytree(ROOT / "i18n/locale", self.folder / "i18n/locale")
        features = self.folder / "logs/smoke/3_feature256"
        features.mkdir(parents=True)
        vectors = np.random.default_rng(0).normal(size=(80, 256)).astype(np.float32)
        np.save(features / "0_0.npy", vectors)
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        environment["PYTHONPATH"] = str(ROOT)
        process = subprocess.run(
            [sys.executable, "-m", "train.train_index", "smoke", "v1",
             str(self.folder / "indices"), "1", "single"],
            cwd=self.folder, env=environment, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        indexes = list((self.folder / "logs/smoke").glob("added_*.index"))
        self.assertEqual(len(indexes), 1)
        index = faiss.read_index(str(indexes[0]))
        self.assertEqual(index.ntotal, 80)
        distances, matches = index.search(vectors[:1], 1)
        self.assertGreaterEqual(matches[0, 0], 0)
        self.assertLess(distances[0, 0], 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
