"""
Sweep the decision threshold and score P/R/F1 at every operating point.

  python threshold_sweep.py --adapter runs/V2_w2048/adapter --split test

Greedy decoding makes the break/no-break call a hard argmax on ONE token: if
P(NONE) beats P("1") by 0.01 the window emits nothing. Measured on test, 10 of
16 documents came back completely silent, losing 44 of 64 breaks — while the
documents it did engage scored 4/4, 4/4, 3/3. The judgment is sound; the
operating point is not.

THE TRICK THAT MAKES THIS CHEAP: generation runs ONCE per window with the
decision forced to "1", so every window produces its anchors regardless of what
greedy would have chosen. The threshold then only decides which windows' outputs
to KEEP, which is free. So an arbitrary number of thresholds costs one
generation pass, not one pass each.

Greedy still generates the anchor text itself — deterministic verbatim quoting
is what lets a span be located back to a word offset (architecture.md §5's
actual concern). Only the decision token is thresholded.
"""
import argparse
import json
import pathlib

import torch

import config as C
import build_dataset as B
import infer as I
import evaluate as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--base_model",
                    default=str(C.PREPARED_MODEL) if C.PREPARED_MODEL.exists()
                    else C.MODEL_NAME)
    ap.add_argument("--gen_batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_docs", type=int, default=None,
                    help="score only the first N documents of the split. The "
                         "eval was taking LONGER than training on the full val "
                         "set (480 windows); 16 docs is ~70 windows and is the "
                         "single biggest speedup for screening runs.")
    ap.add_argument("--dump_at", type=float, default=None,
                    help="also write a predictions jsonl at this threshold, in "
                         "the same format infer.py produces, so downstream "
                         "tooling can consume a chosen operating point")
    ap.add_argument("--thresholds",
                    default="0.5,0.4,0.3,0.2,0.15,0.1,0.05,0.02,0.01,0.005,0.0")
    args = ap.parse_args()

    I.apply_saved_geometry(args.adapter)
    model, tokenizer = I.load_model(args.adapter, args.base_model)
    device = next(model.parameters()).device
    yes_id = tokenizer("1", add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer(C.NO_BREAK_TARGET, add_special_tokens=False)["input_ids"][0]

    transcripts, annotations = B.load_sources()
    annotations = B.validate(transcripts, annotations)
    gold = {a["record_id"]: sorted(a["breaks"]) for a in annotations}
    wanted = {r["record_id"] for r in
              B.load_jsonl(C.BUILD_DIR / f"dataset_{args.split}.jsonl")}
    docs = [d for d in B.load_jsonl(C.TRANSCRIPTS) if d["record_id"] in wanted]
    if args.max_docs:
        docs = sorted(docs, key=lambda d: d["record_id"])[:args.max_docs]

    pool = []
    for d in docs:
        words = d["body"].split()
        for w in I.enumerate_windows(words, tokenizer):
            pool.append({**w, "rid": d["record_id"], "words": words})
    print(f"{len(docs)} documents, {len(pool)} windows\n")

    tokenizer.padding_side = "left"
    order = sorted(range(len(pool)), key=lambda i: -len(pool[i]["prompt"]))

    for bs in range(0, len(order), args.gen_batch_size):
        idxs = order[bs:bs + args.gen_batch_size]
        # Force the decision to "1" so every window yields anchors; the
        # threshold decides later which to keep.
        prompts = [pool[i]["prompt"] + "1" for i in idxs]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        # Two passes, freed between them. Holding the plain-forward logits
        # (batch x seq x 128k vocab) while generate() allocates its KV cache is
        # what OOM'd the 3072-token sweep at batch 16.
        with torch.no_grad():
            plain = tokenizer([pool[i]["prompt"] for i in idxs],
                              return_tensors="pt", padding=True).to(device)
            logits_last = model(**plain).logits[:, -1, :].float()
            probs = torch.softmax(logits_last, -1).cpu()
            del plain, logits_last
            torch.cuda.empty_cache()
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, num_beams=1,
                                 pad_token_id=tokenizer.pad_token_id)

        gen = out[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idxs):
            item = pool[i]
            item["p_yes"] = probs[j, yes_id].item()
            item["p_no"] = probs[j, no_id].item()
            text = "1" + tokenizer.decode(gen[j], skip_special_tokens=False)
            local = []
            for pre, post in I.parse_anchors(text):
                loc = I.localize(item["window_words"], pre, post)
                if loc is None:
                    continue
                if I.in_guard_zone(loc, len(item["window_words"]),
                                   item["is_first"], item["is_last"]):
                    continue
                local.append(item["w0"] + loc)
            item["breaks"] = local
        if (bs // args.gen_batch_size) % 5 == 0:
            print(f"  generated {min(bs+args.gen_batch_size, len(order))}/"
                  f"{len(order)} windows", flush=True)

    print(f"\n{'threshold':>10}{'windows':>9}{'preds':>7}{'tp':>5}{'fn':>5}"
          f"{'fp':>5}{'P':>7}{'R':>7}{'F1':>8}{'WD':>7}")
    print("-" * 70)

    rows = []
    for t in [float(x) for x in args.thresholds.split(",")]:
        by_doc = {}
        fired = 0
        for it in pool:
            if it["p_yes"] > t:
                fired += 1
                by_doc.setdefault(it["rid"], []).extend(it["breaks"])
        tp = fn = fp = 0
        wds = []
        npred = 0
        for d in docs:
            rid = d["record_id"]
            pred = I.dedupe(by_doc.get(rid, []), C.MATCH_TOLERANCE_WORDS)
            npred += len(pred)
            g = gold[rid]
            m, miss, extra = E.match(g, pred, C.MATCH_TOLERANCE_WORDS)
            tp += len(m); fn += miss; fp += extra
            _, wd = E.pk_and_windowdiff(g, pred, len(d["body"].split()))
            if wd is not None:
                wds.append(wd)
        p, r, f1 = E.prf(tp, fn, fp)
        wd = sum(wds) / len(wds) if wds else float("nan")
        rows.append((t, fired, npred, tp, fn, fp, p, r, f1, wd))
        print(f"{t:>10.3f}{fired:>9}{npred:>7}{tp:>5}{fn:>5}{fp:>5}"
              f"{p:>7.3f}{r:>7.3f}{f1:>8.4f}{wd:>7.3f}")

    # greedy = what the model does today
    g_fired = sum(1 for it in pool if it["p_yes"] > it["p_no"])
    best = max(rows, key=lambda r: r[8])
    print(f"\ngreedy (argmax) fires on {g_fired}/{len(pool)} windows "
          f"({100*g_fired/len(pool):.1f}%)")
    print(f"BEST F1 {best[8]:.4f} at threshold {best[0]:.3f}  "
          f"(P {best[6]:.3f} R {best[7]:.3f}, WD {best[9]:.3f})")
    print(f"RESULT\t{pathlib.Path(args.adapter).parent.name}\t{best[8]:.4f}\t"
          f"{best[0]:.3f}\t{best[6]:.3f}\t{best[7]:.3f}\t{best[9]:.3f}")

    tag = pathlib.Path(args.adapter).parent.name

    if args.dump_at is not None:
        by_doc = {}
        for it in pool:
            if it["p_yes"] > args.dump_at:
                by_doc.setdefault(it["rid"], []).extend(it["breaks"])
        dpath = C.BUILD_DIR / f"pred_{tag}_tau{args.dump_at:g}_{args.split}.jsonl"
        with open(dpath, "w", encoding="utf-8") as f:
            for d in docs:
                rid = d["record_id"]
                f.write(json.dumps({
                    "record_id": rid, "outlet": d.get("outlet"),
                    "date": d.get("date"),
                    "word_count": len(d["body"].split()),
                    "pred_breaks": I.dedupe(by_doc.get(rid, []),
                                            C.MATCH_TOLERANCE_WORDS),
                    "n_unlocatable_anchors": 0, "threshold": args.dump_at}) + "\n")
        print(f"wrote predictions at tau={args.dump_at:g} -> {dpath}")

    out = C.BUILD_DIR / f"threshold_sweep_{tag}_{args.split}.json"
    json.dump([dict(zip(("threshold", "windows_firing", "n_pred", "tp", "fn",
                         "fp", "precision", "recall", "f1", "windowdiff"), r))
               for r in rows], open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
