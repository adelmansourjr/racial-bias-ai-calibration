"""Resolve historical study sources from this repository checkout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.append(str(WORKSPACE / "nameguard-witness-study"))
sys.path.append(str(WORKSPACE / "nameguard-utility-study"))
