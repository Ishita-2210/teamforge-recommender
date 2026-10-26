"""Portable location for local CSVs, trusted model artifacts and runtime state."""
import os
from pathlib import Path

BASE = str(Path(os.environ.get("TEAMFORGE_DATA_DIR", Path(__file__).resolve().parent)).expanduser().resolve())
