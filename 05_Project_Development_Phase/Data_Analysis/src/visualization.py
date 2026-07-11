"""Visualization helpers for consistent figure formatting and saving."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def finalize_and_save(path: Path, dpi: int = 300) -> None:
    """Apply common formatting and persist current matplotlib figure."""
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
