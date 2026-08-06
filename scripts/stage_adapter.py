#!/usr/bin/env python3
"""Stage a publishable copy of an adapter for the Hugging Face Hub.

    python scripts/stage_adapter.py SOURCE_ADAPTER PREPARED_BASE OUT_DIR

Copies the adapter, then applies the two changes it needs before it can leave
this machine:

1. **Rewrites `base_model_name_or_path`.** Training recorded an absolute local
   path. Published as-is, every downstream load either fails or silently falls
   back, so it is rewritten to the Hub id of the ungated base.
2. **Writes `story_break_rows.safetensors`** -- 16 KB holding the two absolute
   `<|STORY_BREAK|>` embedding rows.

On (2): the adapter stores that token as a *delta*, so the correct final row is
`base_row + delta` and the base row has to be the one training saw. Rebuilding
it with `resize_token_embeddings(mean_resizing=True)` reconstructs it to a
relative ~2e-3, which is at or below bf16 noise, so it works today. But that
holds only because transformers scales the covariance by `epsilon = 1e-9`
before sampling, making the "random" draw essentially the mean embedding
vector. Removing that epsilon would be a defensible upstream fix, and the row
would then be a free draw. Shipping the rows costs 16 KB and removes the
dependency -- and, because loading them lets the resize run with
`mean_resizing=False`, it also skips a ~50 s covariance on every model load.

The source adapter is never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from breakingnews.config import DEFAULT_BASE_MODEL, ROWS_FILENAME
from breakingnews.loading import export_break_token_rows, verify_adapter


def main() -> int:
    """Entry point.

    Returns:
        0 if the staged adapter passes verification, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("adapter", type=Path)
    ap.add_argument("prepared_base", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--base-id", default=DEFAULT_BASE_MODEL)
    args = ap.parse_args()

    if args.out.exists():
        sys.exit(f"{args.out} already exists; remove it or pick another path")
    shutil.copytree(args.adapter, args.out)
    print(f"copied {args.adapter} -> {args.out}")

    cfg_path = args.out / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    was = cfg.get("base_model_name_or_path")
    cfg["base_model_name_or_path"] = args.base_id
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"base_model_name_or_path: {was}\n                      -> {args.base_id}")

    rows = export_break_token_rows(args.prepared_base, args.out / ROWS_FILENAME)
    print(f"wrote {rows.name} ({rows.stat().st_size:,} bytes)")

    problems = verify_adapter(args.out)
    if problems:
        print(f"\nSTILL {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nverify_adapter: OK -- ready to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
