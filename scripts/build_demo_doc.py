"""Render the demo upload document (not part of the seeded corpus) for the screen recording.

    python scripts/build_demo_doc.py   ->  sample_data/demo/nellore_coastal_plan.pdf
"""
from __future__ import annotations

from build_sample_docs import OUT, RENDERERS, parse_source

SRC = OUT / "demo" / "nellore_coastal_plan.md"

if __name__ == "__main__":
    doc = parse_source(SRC)
    out = OUT / doc.front["output"]
    RENDERERS[out.suffix](doc, out)
    print(f"built {out.relative_to(OUT.parent)} ({out.stat().st_size:,} bytes)")
