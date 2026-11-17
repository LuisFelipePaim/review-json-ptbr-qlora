"""Registra versões transitivas sem incluir caminhos locais de instalação editable."""

import importlib.metadata
import re
import sys
from pathlib import Path

if sys.platform != "win32" or sys.version_info[:2] != (3, 10):
    raise SystemExit("Este lock descreve o perfil Windows com Python 3.10.")
if sys.prefix == sys.base_prefix:
    raise SystemExit("Execute com .venv/Scripts/python.exe para preservar o ambiente global.")

EXCLUDE = {"pip", "setuptools", "wheel", "review-json-ptbr-qlora"}
packages = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata["Name"]
    canonical = re.sub(r"[-_.]+", "-", name).lower()
    if canonical not in EXCLUDE:
        packages[canonical] = distribution.version

header = (
    "# Gerado por scripts/freeze_environment.py no ambiente de referencia Windows.\n"
    "# Instale com scripts/setup_windows.ps1 -UseLock (inclui a wheel CUDA oficial).\n"
    "# Ferramentas de bootstrap estao fixadas em setup_windows.ps1 e pyproject.toml.\n"
)
path = Path(__file__).resolve().parents[1] / "requirements-windows.lock.txt"
path.write_text(
    header + "".join(f"{name}=={version}\n" for name, version in sorted(packages.items())),
    encoding="utf-8",
)
print(f"{len(packages)} distribuicoes registradas em {path.name}.")
