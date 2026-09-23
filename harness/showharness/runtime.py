"""Locate the bundled upstream Show-Harness modules and prompt assets."""
from pathlib import Path
import sys

UPSTREAM_ROOT = Path(__file__).resolve().parent / "upstream"


def ensure_upstream():
    """Expose upstream's absolute core/plugins imports once, independent of cwd."""
    path = str(UPSTREAM_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return UPSTREAM_ROOT
