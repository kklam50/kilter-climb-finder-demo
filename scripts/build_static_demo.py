"""
Assemble the deployable static site into static_demo/:
client/* + data/ (from export_static_demo.py) + a Hugging Face static-Space README.

Usage: python scripts/build_static_demo.py [--data data] [--out static_demo]
"""

import argparse
import shutil
from pathlib import Path

SPACE_README = """---
title: Kilter Climb Finder (precomputed demo)
emoji: 🧗
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
---

# Kilter Climb Finder — precomputed demo

Results here are stored output from an offline export of the Kilter Climb
Finder retrieval engine, not live search.
"""

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="static_demo")
args = ap.parse_args()

root = Path(__file__).resolve().parent.parent
out = Path(args.out)
if out.exists():
    shutil.rmtree(out)
shutil.copytree(root / "client", out)
shutil.copytree(args.data, out / "data")
(out / "README.md").write_text(SPACE_README, encoding="utf-8")
print(f"built {out}")
