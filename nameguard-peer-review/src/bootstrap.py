"""Resolve immutable sibling-study dependencies without importing MLX."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0,str(WORKSPACE/'native-token-gate-hardening'))
sys.path.insert(0,str(WORKSPACE/'nameguard-validation/src'))
