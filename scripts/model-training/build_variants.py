"""
Pre-build the dataset grid the HPO search selects over.

  python build_variants.py            # builds build/variants/<id>/
  python build_variants.py --report   # grid summary only, nothing written

Window/stride/guard are DATA parameters — changing one means rebuilding the
dataset, which cannot happen inside an Optuna trial without paying the
tokenization cost every time.  So the grid is built once up front and the search
treats `variant_id` as a single categorical dimension.

The expensive part of a build is mapping words to tokens over 484 transcripts
(~4 min).  That map does not depend on window/stride/guard at all, so it is
computed ONCE and reused across every variant, which turns an 18 x 4 min job
into 4 min plus seconds per variant.

Splits stay grouped by record_id with the same seed across all variants, so a
transcript lands in the same split everywhere and trial scores stay comparable.
"""
import argparse
import bisect
import itertools
import json
import shutil
from collections import Counter

import config as C
import build_dataset as B

# The grid. Guard is absolute (tokens of speech runway), not a fraction of the
# window, because the quantity that matters is how much broadcast the model gets
# to hear before committing — that does not scale with our slicing choice.
WINDOW_TOKENS = [2048, 3072, 4096]
STRIDE_RATIO = [0.50, 0.75]
EDGE_GUARD_LO = [200, 300, 400]


def variant_id(window, ratio, guard):
    return f"w{window}_s{int(window * ratio)}_g{guard}"


def build_one(window, ratio, guard, cache, transcripts, annotations, tokenizer):
    """Build one variant by mutating config, reusing the cached word->token maps."""
    C.WINDOW_TOKENS = window
    C.STRIDE_TOKENS = int(window * ratio)
    C.EDGE_GUARD_LO = guard
    C.EDGE_GUARD_HI = window - guard
    C.MIN_WINDOW_TOKENS = window // 4
    C.MAX_SEQ_LEN = window + 1024          # window + prompt scaffold + anchors

    rows, rejects = [], []
    for a in annotations:
        rec = transcripts[a["record_id"]]
        words, w2t, n_tokens = cache[a["record_id"]]
        breaks = a["breaks"]
        break_tok = {b: w2t[b] for b in breaks}
        all_break_tok = [break_tok[b] for b in breaks]

        for grid_t0 in B.window_starts(n_tokens):
            grid_t1 = min(grid_t0 + window, n_tokens)
            t0, t1, category, note = B.clip_to_guard(grid_t0, grid_t1, all_break_tok)
            w0 = bisect.bisect_left(w2t, t0)
            w1 = bisect.bisect_left(w2t, t1)
            in_window = [b for b in breaks if w0 < b < w1]

            base = {
                "record_id": a["record_id"], "outlet": a["outlet"],
                "date": a["date"], "show": a["show"], "type": rec.get("type"),
                "ann_category": a["category"],
                "grid_tok_start": grid_t0, "grid_tok_end": grid_t1,
                "win_tok_start": t0, "win_tok_end": t1,
                "win_word_start": w0, "win_word_end": w1,
                "was_clipped": note == "clipped",
                "breaks_global": in_window,
                "breaks_local_tok": [break_tok[b] - t0 for b in in_window],
            }
            if category is None or w1 <= w0:
                rejects.append({**base, "reject_reason": note or "empty_word_span"})
                continue

            window_words = words[w0:w1]
            local_words = [b - w0 for b in in_window]
            rows.append({
                **base,
                "input_text": " ".join(window_words),
                "target_text": B.build_target(window_words, local_words),
                "category": category,
                "breaks_local_word": local_words,
                "n_breaks": len(local_words),
                "n_anchors_unlocatable": 0,
            })

    B.assign_splits(rows, annotations)
    return rows, rejects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_root = C.BUILD_DIR / "variants" if args.out is None else __import__(
        "pathlib").Path(args.out)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME)

    transcripts, annotations = B.load_sources()
    annotations = B.validate(transcripts, annotations)
    gold = sum(len(a["breaks"]) for a in annotations)

    print(f"caching word->token maps for {len(annotations)} transcripts "
          f"(the slow part, done once)...")
    cache = {}
    for a in annotations:
        words = transcripts[a["record_id"]]["body"].split()
        w2t, n_tok = B.word_to_token_index(words, tokenizer)
        cache[a["record_id"]] = (words, w2t, n_tok)
    print("cached.\n")

    grid = list(itertools.product(WINDOW_TOKENS, STRIDE_RATIO, EDGE_GUARD_LO))
    print(f"{'variant':<22}{'rows':>6}{'train':>7}{'val':>6}{'test':>6}"
          f"{'CatA%':>7}{'covered':>16}{'max_len':>8}")
    print("-" * 88)

    manifest = []
    for window, ratio, guard in grid:
        vid = variant_id(window, ratio, guard)
        rows, rejects = build_one(window, ratio, guard, cache,
                                  transcripts, annotations, tokenizer)
        counts = Counter(r["split"] for r in rows)
        cov = len({(r["record_id"], b) for r in rows for b in r["breaks_global"]})
        cat_a = sum(1 for r in rows if r["category"] == "true_transition")

        entry = {
            "variant_id": vid, "window_tokens": window,
            "stride_tokens": int(window * ratio), "stride_ratio": ratio,
            "edge_guard_lo": guard, "max_seq_len": window + 1024,
            "n_rows": len(rows), "n_rejected": len(rejects),
            "n_train": counts["train"], "n_val": counts["val"],
            "n_test": counts["test"],
            "cat_a_share": cat_a / max(1, len(rows)),
            "breaks_covered": cov, "coverage": cov / gold,
        }
        manifest.append(entry)
        print(f"{vid:<22}{len(rows):>6}{counts['train']:>7}{counts['val']:>6}"
              f"{counts['test']:>6}{100*cat_a/max(1,len(rows)):>6.1f}%"
              f"{cov:>7} ({100*cov/gold:>5.1f}%){window+1024:>8}")

        if args.report:
            continue
        vdir = out_root / vid
        vdir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            with open(vdir / f"dataset_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    if r["split"] == split:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(vdir / "meta.json", "w") as f:
            json.dump(entry, f, indent=2)

    print("-" * 88)
    print(f"{len(grid)} variants, gold breaks {gold}")

    if not args.report:
        with open(out_root / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        size = sum(f.stat().st_size for f in out_root.rglob("*.jsonl"))
        print(f"\nwrote {out_root}  ({size/1e6:.0f} MB)")
        print(f"manifest -> {out_root/'manifest.json'}")


if __name__ == "__main__":
    main()
