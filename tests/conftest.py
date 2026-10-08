"""Load only modules from this standalone project during verification."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for relative in (
    ".",
    "data_processing",
    "datasets",
    "rl_envs",
    "segmentation",
    "simulation",
    "meta_rl",
):
    path = str(ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)
