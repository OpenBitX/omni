#!/usr/bin/env python3
"""Idempotent model downloader.

Downloads model files needed by omni-backend into backend/models/:
  - yolo11s-seg.pt        (YOLO segmentation, ~22 MB)

Usage:
    uv run python scripts/download_models.py
    # or via npm:
    pnpm backend:models

Each model is only downloaded if the destination file is absent.
Re-run safely any time.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = [
    {
        "name": "yolo11s-seg.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
        "dest": MODELS_DIR / "yolo11s-seg.pt",
        "compressed": False,
        "min_bytes": 20_000_000,
    },
]


class _ProgressReporter:
    def __init__(self, name: str):
        self.name = name
        self._last_pct = -1

    def __call__(self, count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        pct = min(100, int(count * block_size * 100 / total_size))
        if pct != self._last_pct and pct % 10 == 0:
            print(f"  {self.name}: {pct}%", flush=True)
            self._last_pct = pct


def _download(url: str, dest: Path, compressed: bool, min_bytes: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        print(f"  downloading from {url}")
        urllib.request.urlretrieve(url, str(tmp), reporthook=_ProgressReporter(dest.name))
        tmp.rename(dest)
        actual = dest.stat().st_size
        if actual < min_bytes:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"downloaded file too small ({actual:,} bytes < {min_bytes:,} expected) — "
                "possibly a partial or error page"
            )
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main() -> None:
    errors: list[str] = []
    for spec in MODELS:
        dest: Path = spec["dest"]
        name: str = spec["name"]
        if dest.exists():
            size = dest.stat().st_size
            if size >= spec["min_bytes"]:
                print(f"OK  {name} already present ({size / 1_048_576:.1f} MB)")
                continue
            else:
                print(
                    f"WARN {name} exists but looks incomplete ({size:,} B < {spec['min_bytes']:,} B) — re-downloading"
                )
                dest.unlink()

        print(f"GET {name}")
        try:
            _download(spec["url"], dest, spec["compressed"], spec["min_bytes"])
            print(f"OK  {name} ({dest.stat().st_size / 1_048_576:.1f} MB)")
        except Exception as e:
            msg = f"ERR {name} FAILED: {e}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    if errors:
        print("\nThe following models failed to download:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll models ready in", MODELS_DIR)


if __name__ == "__main__":
    main()
