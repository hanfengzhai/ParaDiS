#!/usr/bin/env python3
"""Inline local <img src="..."> paths in Marp HTML as base64 data URIs."""
import argparse
import base64
import mimetypes
import re
import sys
from pathlib import Path

IMG_RE = re.compile(r'(<img\s+[^>]*\bsrc=")([^"]+)(")')


def embed_images(html_path: Path) -> int:
    html = html_path.read_text(encoding="utf-8")
    base = html_path.parent
    embedded = 0
    missing = 0

    def replace(match: re.Match) -> str:
        nonlocal embedded, missing
        prefix, src, suffix = match.group(1), match.group(2), match.group(3)
        if src.startswith(("data:", "http://", "https://", "//")):
            return match.group(0)
        img_path = (base / src).resolve()
        if not img_path.is_file():
            missing += 1
            print(f"WARN missing image: {img_path}", file=sys.stderr)
            return match.group(0)
        mime = mimetypes.guess_type(img_path.name)[0] or "image/png"
        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        embedded += 1
        return f'{prefix}data:{mime};base64,{data}{suffix}'

    updated = IMG_RE.sub(replace, html)
    if embedded:
        html_path.write_text(updated, encoding="utf-8")
    print(f"Embedded {embedded} image(s) in {html_path}")
    if missing:
        print(f"Missing {missing} image(s)", file=sys.stderr)
    return 0 if not missing else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path, help="Marp HTML output to patch in place")
    args = parser.parse_args()
    if not args.html.is_file():
        print(f"Not found: {args.html}", file=sys.stderr)
        return 1
    return embed_images(args.html)


if __name__ == "__main__":
    raise SystemExit(main())
