"""Diagnóstico local sem carregar pesos ou coletar variáveis de ambiente."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGES = (
    "torch", "transformers", "peft", "bitsandbytes", "accelerate",
    "huggingface-hub", "hf-xet", "numpy", "psutil", "safetensors", "tokenizers",
)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def environment_report() -> dict[str, Any]:
    import psutil
    import torch

    memory = psutil.virtual_memory()
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15, check=True,
            )
            report["nvidia_smi"] = result.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            report["nvidia_smi_error"] = str(error)
    if report["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        report["gpu"] = {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_bytes": total,
            "free_bytes": free,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    missing = [name for name, version in report["packages"].items() if version is None]
    report["missing_packages"] = missing
    report["status"] = (
        "missing_packages" if missing else "ok" if report["cuda_available"] else "cuda_unavailable"
    )
    if report["status"] == "ok":
        try:
            import bitsandbytes as bnb
            import peft

            # Um forward de camada pequena testa a DLL CUDA antes de baixar o LLM.
            # Não há otimizador, backward ou acesso a dados.
            with torch.inference_mode():
                layer = bnb.nn.Linear4bit(
                    64, 32, bias=False, compute_dtype=torch.float16,
                    compress_statistics=True, quant_type="nf4",
                ).to(device="cuda:0", dtype=torch.float16)
                output = layer(torch.ones(1, 64, device="cuda:0", dtype=torch.float16))
                if not torch.isfinite(output).all().item():
                    raise RuntimeError("O forward NF4 produziu valores não finitos.")
                report["nf4_cuda_forward"] = {
                    "status": "ok", "output_shape": list(output.shape),
                    "quant_type": layer.weight.quant_state.quant_type,
                    "storage_dtype": str(layer.weight.dtype),
                    "peft_import_ok": bool(peft.__version__),
                }
            del layer, output
            torch.cuda.empty_cache()
        except Exception as error:
            report["status"] = "nf4_probe_failed"
            report["nf4_cuda_forward"] = {
                "status": "error", "error_type": type(error).__name__, "error": str(error)
            }
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/local/environment.json"))
    args = parser.parse_args()
    print("Verificando bibliotecas, CUDA e um forward NF4 pequeno...", flush=True)
    try:
        report = environment_report()
    except Exception as error:
        report = {"status": "error", "error_type": type(error).__name__, "error": str(error)}
    write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

