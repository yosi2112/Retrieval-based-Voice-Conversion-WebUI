# 現行Google Colabでの実行

2026年10月9日時点のGoogle公式情報では、標準ランタイムはPython 3.13です。
2026年9月16日の更新でPython 3.12から3.13、Ubuntu 22.04から24.04へ移行しています。
公開パッケージ一覧にはPyTorch/TorchAudio 2.11、NumPy 2.1.3、Librosa 0.11、
Gradio 6.29、Transformers 5.18が掲載されています。
公開一覧はCPUイメージの情報を含み、実際のGPUランタイムとはCUDAビルドが異なる場合があります。

情報源:

- [Google Colab公式更新履歴](https://colab.research.google.com/notebooks/relnotes.ipynb)
- [Google公式のランタイム構成](https://github.com/googlecolab/backend-info)
- [Python 3.13への移行告知](https://github.com/googlecolab/colabtools/issues/6081)

## 実行手順

ColabでGPUランタイムを選択し、[RVC_Colab.ipynb](../../RVC_Colab.ipynb)を開いて順に実行します。
既存ノートブックを使う場合は、更新したリポジトリ直下で次を実行します。

```python
import subprocess
import sys

print(sys.version)
subprocess.run([sys.executable, "tools/colab_setup.py"], check=True)
```

セットアップはColabと同じPythonで`.venv-colab`を作成し、標準環境のPyTorch・TorchAudio・
TorchVision・NumPy・SciPy・Numba・LLVMを引き継ぎます。インストール時には実際の版を
制約ファイルに保存するため、依存解決がCUDA版PyTorchを置き換えることを防ぎます。
追加パッケージは仮想環境へインストールされます。

```python
from pathlib import Path
import subprocess

python = str(Path(".venv-colab/bin/python").resolve())
subprocess.run([python, "tools/colab_check.py", "--require-cuda"], check=True)
subprocess.run([python, "webui.py", "--colab", "--pycmd", python], check=True)
```

WebUIの子プロセスも同じ仮想環境を使用します。現在の環境の詳細は
`.venv-colab/colab-runtime.json`、インストール結果は`install-report.json`へ保存されます。
依存解決だけを確認する場合は`tools/colab_setup.py --dry-run`を使用できます。

## 旧ノートブックから移行するとき

- `python3.12`の追加インストール、Torch 2.7.1/cu118の固定、`numpy<2`の固定を外します。
- `requirments_*_py312.txt`の代わりに`requirements_colab.txt`を使用します。
- Gradio 3/FastAPI 0.99/Pydantic 1の固定や、CUDA/cuDNNパッケージの個別置換を外します。
- `train.*`を`data_utils`などの裸のインポートへ置換するパッチを外します。
  `train/__init__.py`とプロジェクト直下を優先する起動コードをそのまま使用します。
- 旧`.venv`はそのまま残せます。新しい起動先は`.venv-colab/bin/python`です。

TorchAudio 2.9以降の`load()`はTorchCodecに依存します。本コードではFFmpegでデコードし、
TorchAudioのCUDAリサンプリングを利用するため、TorchCodecの追加インストールは不要です。
RVCのモデル形式と既存の学習データ形式は変更しません。

## 検証の範囲

`colab_check.py`は主要ライブラリと学習・推論モジュールの読み込み、GradioのUI構築、
メルスペクトログラムの順伝播・逆伝播を確認します。
モデルをダウンロード済みなら`--require-models`を追加するとHuBERTの読み込みも確認します。
このチェックはデータセットを使った学習全体の完走を保証するものではありません。
