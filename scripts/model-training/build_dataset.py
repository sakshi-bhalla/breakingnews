"""
Stage 1 — build the LoRA training set from annotated transcripts.

This is the stitching/windowing step `README.md` flagged as missing.  Per
decision D2 it windows the annotated dumps directly rather than stitching
synthetic story pairs: the 484 annotations already carry 1,375 real in-situ
broadcast transitions, and a synthetic seam is a shortcut the model can learn
instead of learning editorial structure.

    python build_dataset.py                # writes build/
    python build_dataset.py --report-only  # stats, no files written

Pipeline per transcript:
  words -> exact word->token map (Llama tokenizer, offset mapping)
        -> slide WINDOW_TOKENS at STRIDE_TOKENS
        -> route each window through the edge-guard table (architecture.md §3)
        -> emit Cat A (has breaks) / Cat C (clean) rows in anchor format (D1)

NOTHING is discarded silently.  Every rejected window is written to
build/rejected_windows.jsonl with its reason so the edge guard is auditable.
"""
import argparse
import bisect
import json
import random
from collections import Counter, defaultdict

import config as C


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sources():
    """
    Return (transcripts_by_id, annotations) with the heldout flag attached.

    Heldout status comes from HELDOUT_ID_SOURCES (a set of record_ids), not from
    which file an annotation arrived in — the verified transcripts are now also
    present in the full annotation pass, so a per-file flag would mark them
    trainable and leak the test set.
    """
    transcripts = {r["record_id"]: r for r in load_jsonl(C.TRANSCRIPTS)}

    heldout_ids = set()
    for path in getattr(C, "HELDOUT_ID_SOURCES", []):
        if path.exists():
            heldout_ids |= {a["record_id"] for a in load_jsonl(path)}

    annotations, seen = [], set()
    for path, file_heldout in C.ANNOTATIONS:
        for a in load_jsonl(path):
            rid = a["record_id"]
            if rid in seen:            # later files never override earlier ones
                continue
            seen.add(rid)
            a["_heldout"] = file_heldout or (rid in heldout_ids)
            annotations.append(a)
    return transcripts, annotations


def validate(transcripts, annotations):
    """Fail loudly on any annotation we cannot trust. Returns the clean list."""
    ok, problems = [], []
    for a in annotations:
        rid = a["record_id"]
        if rid not in transcripts:
            problems.append((rid, "no matching transcript"))
            continue
        words = transcripts[rid]["body"].split()
        if len(words) != a["word_count"]:
            problems.append((rid, f"word_count {a['word_count']} != body {len(words)}"))
            continue
        bad = [b for b in a["breaks"] if not (0 < b < len(words))]
        if bad:
            problems.append((rid, f"break offsets out of range: {bad}"))
            continue
        if a["breaks"] != sorted(set(a["breaks"])):
            a["breaks"] = sorted(set(a["breaks"]))
        ok.append(a)
    if problems:
        print(f"[WARN] dropped {len(problems)} unusable annotations:")
        for rid, why in problems[:20]:
            print(f"       {rid}: {why}")
    return ok


# ── Word <-> token mapping ─────────────────────────────────────────────────────

def word_to_token_index(words, tokenizer):
    """
    Exact map from word index -> index of the first token covering that word,
    computed on the whitespace-joined text the rows actually carry.

    Returns a list of length len(words)+1; the final entry is the token count,
    so w2t[i:j] slicing behaves like any other half-open range.
    """
    text = " ".join(words)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    # Character offset at which each word starts (words joined by one space).
    starts, cursor = [], 0
    for w in words:
        starts.append(cursor)
        cursor += len(w) + 1

    w2t, tok_i = [], 0
    for char_start in starts:
        # Advance to the first token whose span reaches this character.
        while tok_i < len(offsets) and offsets[tok_i][1] <= char_start:
            tok_i += 1
        w2t.append(min(tok_i, len(offsets)))
    w2t.append(len(offsets))
    return w2t, len(offsets)


# ── Windowing ──────────────────────────────────────────────────────────────────

def window_starts(n_tokens):
    """
    Stride grid over the transcript.  A trailing end-anchored window is appended
    so the tail still gets full-context coverage, gated on
    TAIL_WINDOW_MIN_ADVANCE.  A barely-advanced tail window is NOT merely a
    duplicate: shifting the frame a few hundred tokens lifts tail breaks out of
    the guard band, so the gate trades duplicated rows against coverage rather
    than trimming pure waste (config.py carries the measured curve).
    """
    if n_tokens <= C.WINDOW_TOKENS:
        return [0]
    starts = list(range(0, n_tokens - C.WINDOW_TOKENS + 1, C.STRIDE_TOKENS))
    tail_start = n_tokens - C.WINDOW_TOKENS
    if tail_start - starts[-1] >= C.TAIL_WINDOW_MIN_ADVANCE:
        starts.append(tail_start)
    return starts


def clip_to_guard(t0, t1, break_tokens):
    """
    architecture.md §3's edge guard, applied by moving the window edge rather
    than discarding the window.

    A break inside a guard band cannot simply be left out of the target: the
    target asserts it lists EVERY boundary in the slice, so omitting one
    manufactures a false negative.  Discarding the window instead costs all the
    OTHER breaks it contained — measured at 34% of the gold set.  Pulling the
    edge past the offender leaves a slice that is shorter but completely and
    honestly labelled, and the offender reappears mid-window in the neighbour.

    Iterates because shrinking the window also moves its guard bands inward,
    which can engulf a break that was safe a moment ago.

    Returns (new_t0, new_t1, category, reason); category is None if rejected.
    """
    tail_margin = C.WINDOW_TOKENS - C.EDGE_GUARD_HI       # 400
    clipped = False

    for _ in range(32):
        if t1 - t0 < C.EDGE_GUARD_LO + tail_margin:
            return t0, t1, None, "window_shorter_than_guard_bands"
        if t1 - t0 < C.MIN_WINDOW_TOKENS:
            return t0, t1, None, "clipped_below_min_window"

        lo, hi = t0 + C.EDGE_GUARD_LO, t1 - tail_margin
        lead = [b for b in break_tokens if t0 < b < lo]
        trail = [b for b in break_tokens if hi < b < t1]

        if not lead and not trail:
            inside = [b for b in break_tokens if t0 < b < t1]
            category = "true_transition" if inside else "pure_continuation"
            return t0, t1, category, ("clipped" if clipped else None)

        if lead:
            t0 = max(lead) + C.CLIP_JITTER_TOKENS
            clipped = True
        if trail:
            t1 = min(trail) - C.CLIP_JITTER_TOKENS
            clipped = True

    return t0, t1, None, "clip_did_not_converge"


# ── Anchor-format target (D1) ──────────────────────────────────────────────────

def build_target(window_words, local_break_words):
    """
    Target text = one numbered anchor per break.  The model quotes the words
    straddling the boundary rather than regurgitating the window, which is what
    keeps the sequence at ~4.6k tokens instead of ~8.3k and makes inference
    ~40x cheaper.  The quoted pre-context is what localises the prediction back
    to a word offset at inference time.
    """
    if not local_break_words:
        return C.NO_BREAK_TARGET

    lines = []
    for n, b in enumerate(local_break_words, start=1):
        pre = window_words[max(0, b - C.ANCHOR_PRE_WORDS):b]
        post = window_words[b:b + C.ANCHOR_POST_WORDS]
        lines.append(f"{n}. {' '.join(pre)} {C.STORY_BREAK_TOKEN} {' '.join(post)}")
    return "\n".join(lines)


def locate_anchor(window_words, anchor_pre_text):
    """
    Inverse of build_target's pre-context: find the word offset a quoted anchor
    refers to.  Shared with infer.py so training and inference agree exactly.
    Returns the word index immediately after the match, or None.
    """
    needle = anchor_pre_text.split()
    if not needle:
        return None
    n = len(needle)
    hits = [i + n for i in range(len(window_words) - n + 1)
            if window_words[i:i + n] == needle]
    return hits[0] if len(hits) == 1 else (hits[0] if hits else None)


# ── Row construction ───────────────────────────────────────────────────────────

def build_rows(transcripts, annotations, tokenizer):
    rows, rejects = [], []
    realized_lens = []

    for a in annotations:
        rec = transcripts[a["record_id"]]
        words = rec["body"].split()
        breaks = a["breaks"]

        w2t, n_tokens = word_to_token_index(words, tokenizer)
        break_tok = {b: w2t[b] for b in breaks}

        for grid_t0 in window_starts(n_tokens):
            grid_t1 = min(grid_t0 + C.WINDOW_TOKENS, n_tokens)
            all_break_tok = [break_tok[b] for b in breaks]
            t0, t1, category, note = clip_to_guard(grid_t0, grid_t1, all_break_tok)

            # Token bounds -> word bounds on the same half-open convention.
            w0 = bisect.bisect_left(w2t, t0)
            w1 = bisect.bisect_left(w2t, t1)

            in_window = [b for b in breaks if w0 < b < w1]
            base = {
                "record_id": a["record_id"],
                "outlet": a["outlet"],
                "date": a["date"],
                "show": a["show"],
                "type": rec.get("type"),
                "ann_category": a["category"],
                "grid_tok_start": grid_t0, "grid_tok_end": grid_t1,
                "win_tok_start": t0, "win_tok_end": t1,
                "win_word_start": w0, "win_word_end": w1,
                "was_clipped": note == "clipped",
                "breaks_global": in_window,
                "breaks_local_tok": [break_tok[b] - t0 for b in in_window],
            }

            if category is None or w1 <= w0:
                rejects.append({**base,
                                "reject_reason": note or "empty_word_span"})
                continue

            window_words = words[w0:w1]
            local_words = [b - w0 for b in in_window]
            input_text = " ".join(window_words)
            target_text = build_target(window_words, local_words)

            # Round-trip check: every anchor we emit must be findable again.
            unlocatable = 0
            for b in local_words:
                pre = " ".join(window_words[max(0, b - C.ANCHOR_PRE_WORDS):b])
                if locate_anchor(window_words, pre) != b:
                    unlocatable += 1

            realized_lens.append(
                len(tokenizer(C.PROMPT_TEMPLATE.format(
                    pre=C.ANCHOR_PRE_WORDS, post=C.ANCHOR_POST_WORDS,
                    tok=C.STORY_BREAK_TOKEN, none=C.NO_BREAK_TARGET,
                    input_text=input_text) + target_text,
                    add_special_tokens=True)["input_ids"])
            )

            rows.append({
                **base,
                "input_text": input_text,
                "target_text": target_text,
                "category": category,
                "breaks_local_word": local_words,
                "n_breaks": len(local_words),
                "n_anchors_unlocatable": unlocatable,
            })

    return rows, rejects, realized_lens


# ── Splits ─────────────────────────────────────────────────────────────────────

def assign_splits(rows, annotations):
    """
    Grouped by record_id.  Windows overlap by 1,024 tokens, so a window-level
    split would put the same text on both sides of the train/eval line.

    test = the two independently verified batches (20 transcripts).
    val  = a category-stratified sample of the remaining record_ids.
    """
    heldout = {a["record_id"] for a in annotations if a["_heldout"]}
    pool = [a for a in annotations if not a["_heldout"]]

    by_cat = defaultdict(list)
    for a in pool:
        by_cat[a["category"]].append(a["record_id"])

    rng = random.Random(C.SEED)
    val = set()
    for cat, ids in sorted(by_cat.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        val.update(ids[:max(1, round(len(ids) * C.VAL_FRACTION))])

    for r in rows:
        rid = r["record_id"]
        r["split"] = "test" if rid in heldout else ("val" if rid in val else "train")
    return {"test": sorted(heldout), "val": sorted(val)}


# ── Reporting ──────────────────────────────────────────────────────────────────

def report(rows, rejects, realized_lens, transcripts, annotations):
    print("\n" + "=" * 74)
    print("BUILD SUMMARY")
    print("=" * 74)
    print(f"annotated transcripts in : {len(annotations)}")
    print(f"windows emitted          : {len(rows)}")
    clipped = sum(1 for r in rows if r["was_clipped"])
    print(f"    of which edge-clipped: {clipped} "
          f"({100*clipped/max(1,len(rows)):.1f}%) — a break sat in a guard band, "
          f"so\n                           the window edge moved instead of "
          f"discarding the window")
    widths = sorted(r["win_tok_end"] - r["win_tok_start"] for r in rows)
    print(f"    window width tokens  : min {widths[0]}  "
          f"p50 {widths[len(widths)//2]}  max {widths[-1]}")
    print(f"windows rejected         : {len(rejects)}")
    for reason, n in Counter(r["reject_reason"] for r in rejects).most_common():
        print(f"    {reason:<34} {n}")

    print("\nrows by split x category")
    grid = Counter((r["split"], r["category"]) for r in rows)
    cats = ["true_transition", "pure_continuation"]
    print(f"    {'split':<8}" + "".join(f"{c:>20}" for c in cats) + f"{'total':>10}")
    for split in ("train", "val", "test"):
        cells = [grid[(split, c)] for c in cats]
        print(f"    {split:<8}" + "".join(f"{n:>20}" for n in cells) + f"{sum(cells):>10}")
    tot = [sum(grid[(s, c)] for s in ('train', 'val', 'test')) for c in cats]
    share = [100 * t / max(1, sum(tot)) for t in tot]
    print(f"    {'ALL':<8}" + "".join(f"{n:>20}" for n in tot) + f"{sum(tot):>10}")
    print(f"    {'(share)':<8}" + "".join(f"{s:>19.1f}%" for s in share))
    print("    NOTE: Cat B (hard negatives) is absent by design — per README it "
          "comes\n          from error mining after v1, so 40/30/30 is not "
          "reachable yet.")

    n_breaks = sum(r["n_breaks"] for r in rows)
    gold = sum(len(a["breaks"]) for a in annotations)
    print(f"\nbreaks in emitted windows: {n_breaks} "
          f"(overlap means a break can appear in >1 window)")
    print(f"distinct gold breaks     : {gold}")
    covered = len({(r['record_id'], b) for r in rows for b in r['breaks_global']})
    print(f"distinct breaks covered  : {covered} ({100*covered/gold:.1f}% of gold)")

    bad = sum(r["n_anchors_unlocatable"] for r in rows)
    print(f"anchors not round-trippable: {bad} "
          f"({100*bad/max(1,n_breaks):.2f}% — these are ambiguous 12-grams)")

    if realized_lens:
        realized_lens = sorted(realized_lens)
        p = lambda q: realized_lens[int(q * (len(realized_lens) - 1))]
        print(f"\nrealized sequence length (prompt+target, tokens)")
        print(f"    min {realized_lens[0]}  p50 {p(.5)}  p95 {p(.95)}  "
              f"p99 {p(.99)}  max {realized_lens[-1]}")
        over = sum(1 for x in realized_lens if x > C.MAX_SEQ_LEN)
        verdict = "OK" if over == 0 else f"{over} rows would TRUNCATE"
        print(f"    MAX_SEQ_LEN = {C.MAX_SEQ_LEN} -> {verdict}")

    print("\noutlet mix (rows)")
    for o, n in Counter(r["outlet"] for r in rows).most_common():
        print(f"    {o:<8} {n:>6}")
    print("    NOTE: MSNBC/ABC/CBS are annotated but are NOT in the target "
          "corpus\n          (topic_cluster_test has only cnn + fox-tv).")
    print("=" * 74 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="print stats without writing build/")
    ap.add_argument("--model", default=C.MODEL_NAME,
                    help="tokenizer to window with (must match training)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    transcripts, annotations = load_sources()
    annotations = validate(transcripts, annotations)
    print(f"{len(transcripts)} transcripts, {len(annotations)} usable annotations")

    rows, rejects, realized = build_rows(transcripts, annotations, tokenizer)
    splits = assign_splits(rows, annotations)
    report(rows, rejects, realized, transcripts, annotations)

    if args.report_only:
        print("--report-only: nothing written.")
        return

    C.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        out = C.BUILD_DIR / f"dataset_{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {out}")

    with open(C.BUILD_DIR / "rejected_windows.jsonl", "w", encoding="utf-8") as f:
        for r in rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(C.BUILD_DIR / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"wrote {C.BUILD_DIR / 'rejected_windows.jsonl'} and splits.json")


if __name__ == "__main__":
    main()
