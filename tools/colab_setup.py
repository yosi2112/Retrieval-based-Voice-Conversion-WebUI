"""Install the Colab profile while retaining its working Torch/CUDA stack."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESERVED_PACKAGES = (
    "torch", "torchaudio", "torchvision", "numpy", "scipy", "numba", "llvmlite",
)


def installed_versions():
    result = {}
    for name in PRESERVED_PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def validate_runtime(versions):
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Run this installer inside a current Colab Python 3.13 runtime.")
    for name in ("torch", "torchaudio"):
        version = versions.get(name)
        if version is None:
            raise RuntimeError(f"Colab's preinstalled {name} is missing.")
        numbers = tuple(int(part) for part in version.split("+")[0].split(".")[:2])
        if numbers < (2, 11) or numbers[0] != 2:
            raise RuntimeError(f"Expected Colab {name} 2.11+; found {version}.")
    if versions["torch"].split("+")[0] != versions["torchaudio"].split("+")[0]:
        raise RuntimeError("Preinstalled torch and torchaudio releases do not match.")


def main():
    import venv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=PROJECT_ROOT / ".venv-colab")
    parser.add_argument("--dry-run", action="store_true", help="Resolve dependencies without installing them.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow setup on a Colab CPU runtime.")
    args = parser.parse_args()
    versions = installed_versions()
    validate_runtime(versions)
    environment = args.venv.expanduser().resolve()
    if environment.exists() and any(environment.iterdir()):
        config = environment / "pyvenv.cfg"
        if not config.is_file() or "include-system-site-packages = true" not in config.read_text():
            raise RuntimeError(f"Existing {environment} is not a compatible Colab environment; choose another --venv path.")
    # No ensurepip/apt/Python downgrade is needed. The venv inherits Colab's
    # pip and native ML libraries; additional packages install into the venv.
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(environment)
    python = environment / "bin" / "python"
    runtime = json.loads(subprocess.check_output(
        [str(python), "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"],
        text=True,
    ))
    if runtime != [3, 13]:
        raise RuntimeError(f"Existing environment uses Python {runtime}; choose another --venv path.")
    constraints = environment / "colab-native-constraints.txt"
    constraints.write_text(
        "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n",
        encoding="utf-8",
    )
    snapshot = {
        "python": sys.version,
        "platform": platform.platform(),
        "preserved_packages": versions,
        "python_executable": str(python),
    }
    (environment / "colab-runtime.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps(snapshot, indent=2), flush=True)
    command = [
        str(python), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements_colab.txt"),
        "-c", str(constraints), "--report", str(environment / "install-report.json"),
    ]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if args.dry_run:
        return
    check_command = [str(python), str(PROJECT_ROOT / "tools" / "colab_check.py")]
    if not args.allow_cpu:
        check_command.append("--require-cuda")
    subprocess.run(check_command, cwd=PROJECT_ROOT, check=True)
    print(f"Ready: {python} webui.py --colab --pycmd {python}", flush=True)


if __name__ == "__main__":
    main()
