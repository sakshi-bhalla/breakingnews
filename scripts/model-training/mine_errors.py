"""
Mine a trained model's mistakes — the source of Cat B hard negatives.

  python mine_errors.py --predictions build/pred_B_w2048_s42_val.jsonl

FALSE POSITIVES are the prize.  The model fired where the annotator saw no
boundary, which is the operational definition of architecture.md's
`pseudo_transition_span`: a place where register shifts (new speaker, new
correspondent, location cut, commercial return) but the story does not.  These
are hard negatives that are hard FOR THIS MODEL, which is strictly more useful
than guessing in advance what might confuse it.

FALSE NEGATIVES are the complement: real boundaries it walked past.  Not Cat B,
but they show which transitions are too subtle at current data volume.

Writes both with surrounding context, ready for review.  Nothing is auto-
labelled — the researcher owns every include/exclude call.
"""
import argparse
import json
from collections import Counter

import config as C
import build_dataset as B
import evaluate as E

CONTEXT = 60          # words either side, enough to judge the call


def span(words, i, n=CONTEXT):
    lo, hi = max(0, i - n), min(len(words), i + n)
    return " ".join(words[lo:i]), " ".join(words[i:hi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--tolerance", type=int, default=C.MATCH_TOLERANCE_WORDS)
    ap.add_argument("--out_prefix", default="build/hard")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    transcripts, annotations = B.load_sources()
    annotations = B.validate(transcripts, annotations)
    gold_by_id = {a["record_id"]: a for a in annotations}

    fps, fns = [], []
    for p in B.load_jsonl(args.predictions):
        a = gold_by_id.get(p["record_id"])
        if a is None:
            continue
        words = transcripts[p["record_id"]]["body"].split()
        gold, pred = sorted(a["breaks"]), sorted(p["pred_breaks"])
        matched, _, _ = E.match(gold, pred, args.tolerance)
        hit_g = {g for g, _, _ in matched}
        hit_p = {q for _, q, _ in matched}

        meta = {"record_id": p["record_id"], "outlet": a["outlet"],
                "date": a["date"], "show": a["show"]}
        for q in pred:
            if q not in hit_p:
                before, after = span(words, q)
                fps.append({**meta, "word_offset": q, "kind": "false_positive",
                            "nearest_gold": min((abs(q - g) for g in gold),
                                                default=None),
                            "text_before": before, "text_after": after})
        for g in gold:
            if g not in hit_g:
                before, after = span(words, g)
                fns.append({**meta, "word_offset": g, "kind": "false_negative",
                            "nearest_pred": min((abs(g - q) for q in pred),
                                                default=None),
                            "text_before": before, "text_after": after})

    for name, rows in (("false_positives", fps), ("false_negatives", fns)):
        path = f"{args.out_prefix}_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):>4} -> {path}")

    print(f"\nFALSE POSITIVES {len(fps)}  (-> Cat B hard negatives)")
    print(f"  by outlet: {dict(Counter(r['outlet'] for r in fps))}")
    near = [r["nearest_gold"] for r in fps if r["nearest_gold"] is not None]
    if near:
        near.sort()
        print(f"  distance to nearest real break: median {near[len(near)//2]} "
              f"words, min {near[0]}, max {near[-1]}")
        close = sum(1 for d in near if d <= 100)
        print(f"  {close} of {len(near)} ({100*close/len(near):.0f}%) are within "
              f"100 words of a real break\n     -> near-misses, not wild guesses")

    print(f"\nFALSE NEGATIVES {len(fns)}  (missed real boundaries)")
    print(f"  by outlet: {dict(Counter(r['outlet'] for r in fns))}")

    print(f"\n{'='*74}\nSAMPLE FALSE POSITIVES — the model said 'new story here'\n{'='*74}")
    for r in fps[:args.show]:
        print(f"\n[{r['outlet']} {r['date']} {r['show']}] word {r['word_offset']}"
              f"  (nearest real break: {r['nearest_gold']} words away)")
        print(f"  ...{r['text_before'][-260:]}")
        print(f"  >>> PREDICTED BREAK HERE <<<")
        print(f"  {r['text_after'][:260]}...")


if __name__ == "__main__":
    main()
