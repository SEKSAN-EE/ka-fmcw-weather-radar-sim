#!/usr/bin/env python
"""Build the GitHub Pages site (stdlib only).

    python scripts/build_site.py _site

The page sources are written as HTML fragments (title/style first, then the body markup) so they
can also be published elsewhere; this script wraps them into complete documents:

    _site/index.html                 landing page        <- site/index.html
    _site/signal-chain/index.html    signal chain        <- docs/signal_chain/index.html (+ img/)
    _site/radar-3d/index.html        interactive 3D view <- viz/radar_3d.html
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAD = """<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
"""


def wrap(fragment: str) -> str:
    fragment = re.sub(r'^\s*<meta charset="utf-8">\s*', "", fragment)
    m = re.search(r"^<(div|main|section)\b", fragment, flags=re.M)     # first body element
    head, body = (fragment[: m.start()], fragment[m.start():]) if m else ("", fragment)
    return f"{HEAD}{head}</head>\n<body>\n{body}\n</body>\n</html>\n"


def main(out="_site"):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    pages = {
        out / "index.html": ROOT / "site/index.html",
        out / "signal-chain/index.html": ROOT / "docs/signal_chain/index.html",
        out / "radar-3d/index.html": ROOT / "viz/radar_3d.html",
    }
    for dst, src in pages.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(wrap(src.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"  {src.relative_to(ROOT)} -> {dst}")
    shutil.copytree(ROOT / "docs/signal_chain/img", out / "signal-chain/img")
    shutil.copytree(ROOT / "docs/signal_chain/img", out / "img")          # thumbnails for the landing page
    (out / ".nojekyll").write_text("")
    print(f"built {out}/")


if __name__ == "__main__":
    main(*sys.argv[1:])
