"""
Stage 3 — sliding-window inference (architecture.md §5).

  python infer.py --adapter runs/seg_lora_v1/adapter --split test
  python infer.py --adapter runs/seg_lora_v1/adapter \
                  --input /path/to/cnn_sample.jsonl --out preds_cnn.jsonl

Emits one record per document: the predicted story-break word offsets in the
document's own coordinate system, ready to slice the transcript into stories.

Window mapping is identical to build_dataset.py (same tokenizer, same stride,
same word<->token map), so a break the model saw in training sits at the same
relative position it will see here.
"""
import argparse
import bisect
import json
import re
from pathlib import Path

import torch

import config as C
from build_dataset import load_jsonl, locate_anchor, window_starts, word_to_token_index

ANCHOR_LINE = re.compile(r"^\s*\d+[.)]\s*(.*)$")


# ── Parsing generated text ─────────────────────────────────────────────────────

def parse_anchors(generated: str):
    """
    Pull (pre_context, post_context) pairs out of the model's output.
    Tolerant of missing numbering and of the model trailing off mid-line.
    """
    if C.NO_BREAK_TARGET in generated and C.STORY_BREAK_TOKEN not in generated:
        return []

    anchors = []
    for raw in generated.splitlines():
        line = raw.strip()
        if not line or C.STORY_BREAK_TOKEN not in line:
            continue
        m = ANCHOR_LINE.match(line)
        if m:
            line = m.group(1)
        pre, _, post = line.partition(C.STORY_BREAK_TOKEN)
        pre, post = pre.strip(), post.strip()
        if pre:
            anchors.append((pre, post))
    return anchors


def localize(window_words, pre, post):
    """
    Map a generated anchor back to a word offset inside the window.
    Falls back to the post-context, then to progressively shorter pre-context
    tails, since a slightly paraphrased anchor still usually shares its ending.
    Returns None if the anchor cannot be grounded — those are counted, not
    silently dropped.
    """
    hit = locate_anchor(window_words, pre)
    if hit is not None:
        return hit

    post_words = post.split()
    if post_words:
        n = len(post_words)
        for i in range(len(window_words) - n + 1):
            if window_words[i:i + n] == post_words:
                return i

    tail = pre.split()
    for k in (8, 6, 4):
        if len(tail) >= k:
            hit = locate_anchor(window_words, " ".join(tail[-k:]))
            if hit is not None:
                return hit
    return None


# ── Guard zone + dedup (architecture.md §5) ────────────────────────────────────

def in_guard_zone(local_word, n_words, is_first_window, is_last_window):
    """
    Predictions in the outer 10% of a window are discarded — the model lacks the
    context there to confirm a macro shift, and the neighbouring window covers
    that text from a better position.

    Exception: the document's very first and very last windows have no
    neighbour, so applying the guard there would blind the model to breaks near
    the start and end of the transcript entirely.
    """
    margin = int(0.10 * n_words)
    if not is_first_window and local_word < margin:
        return True
    if not is_last_window and local_word > n_words - margin:
        return True
    return False


def dedupe(offsets, tolerance):
    """Overlapping windows see the same break twice; collapse them to one."""
    merged = []
    for o in sorted(offsets):
        if merged and o - merged[-1][-1] <= tolerance:
            merged[-1].append(o)
        else:
            merged.append([o])
    return [int(round(sum(g) / len(g))) for g in merged]


# ── Model loading ──────────────────────────────────────────────────────────────

def apply_saved_geometry(adapter_path):
    """
    Restore the window/stride/guard the adapter was TRAINED with.

    Without this, a model trained on 2,048-token windows gets run with
    config.py's 4,096 default: it sees slices unlike anything in training and
    the guard-zone logic no longer matches the training routing. Silent, and it
    would look like a bad model rather than a mismatched harness.
    """
    import json as _json
    path = Path(adapter_path) / "segmentation_config.json"
    if not path.exists():
        print(f"[WARN] no segmentation_config.json in {adapter_path}; using "
              f"config.py defaults ({C.WINDOW_TOKENS}/{C.STRIDE_TOKENS})")
        return
    g = _json.load(open(path))
    C.WINDOW_TOKENS = g["window_tokens"]
    C.STRIDE_TOKENS = g["stride_tokens"]
    C.EDGE_GUARD_LO = g["edge_guard_lo"]
    C.EDGE_GUARD_HI = g["window_tokens"] - g["edge_guard_lo"]
    print(f"window geometry from adapter: {C.WINDOW_TOKENS}/{C.STRIDE_TOKENS} "
          f"guard {C.EDGE_GUARD_LO}")


def load_model(adapter_path, base_model):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.padding_side = "left"          # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


# ── Main loop ──────────────────────────────────────────────────────────────────

def enumerate_windows(words, tokenizer):
    """Window one document; returns a list of per-window records (no model)."""
    w2t, n_tokens = word_to_token_index(words, tokenizer)
    starts = window_starts(n_tokens)
    out = []
    for wi, t0 in enumerate(starts):
        t1 = min(t0 + C.WINDOW_TOKENS, n_tokens)
        w0 = bisect.bisect_left(w2t, t0)
        w1 = bisect.bisect_left(w2t, t1)
        if w1 <= w0:
            continue
        window_words = words[w0:w1]
        out.append({
            "w0": w0, "window_words": window_words,
            "is_first": wi == 0, "is_last": wi == len(starts) - 1,
            "prompt": C.PROMPT_TEMPLATE.format(
                pre=C.ANCHOR_PRE_WORDS, post=C.ANCHOR_POST_WORDS,
                tok=C.STORY_BREAK_TOKEN, none=C.NO_BREAK_TARGET,
                input_text=" ".join(window_words)),
        })
    return out


def predict_documents(docs, model, tokenizer, max_new_tokens, device,
                      batch_size=8, progress=False, dump_path=None):
    """
    Batched sliding-window inference over MANY documents at once.

    Generating one window at a time wastes most of the GPU: a 4,096-token
    prefill followed by a few dozen sequential decode steps at batch 1 is almost
    entirely memory-bandwidth bound.  Windows are independent — across documents
    as well as within one — so they all go into one pool, get sorted by length to
    minimise left-padding, and are generated in batches.  Typically 5-10x faster
    than the per-document loop for identical greedy output.

    `docs` is a list of {"words": [...]} (extra keys are preserved).
    Returns a list of (break_offsets, n_unlocatable) aligned with `docs`.
    """
    pool = []
    for di, d in enumerate(docs):
        for w in enumerate_windows(d["words"], tokenizer):
            pool.append({**w, "doc": di})
    if not pool:
        return [([], 0) for _ in docs]

    # Sort long->short: batches are then length-homogeneous, so padding waste is
    # small and the largest batch (worst memory case) is hit first, failing fast
    # instead of after most of the work is done.
    order = sorted(range(len(pool)), key=lambda i: -len(pool[i]["prompt"]))

    results = [([], 0) for _ in docs]
    breaks_by_doc = [[] for _ in docs]
    unloc_by_doc = [0] * len(docs)

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"          # required for batched generation
    try:
        for bs in range(0, len(order), batch_size):
            idxs = order[bs:bs + batch_size]
            prompts = [pool[i]["prompt"] for i in idxs]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,          # greedy: temperature 0, no drift
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1]:]
            for row, i in enumerate(idxs):
                item = pool[i]
                text = tokenizer.decode(gen[row], skip_special_tokens=False)
                if dump_path is not None:
                    with open(dump_path, "a", encoding="utf-8") as _f:
                        _f.write(f"=== doc {item['doc']} @word {item['w0']} ===\n"
                                 f"{text[:600]}\n\n")
                for pre, post in parse_anchors(text):
                    local = localize(item["window_words"], pre, post)
                    if local is None:
                        unloc_by_doc[item["doc"]] += 1
                        continue
                    if in_guard_zone(local, len(item["window_words"]),
                                     item["is_first"], item["is_last"]):
                        continue
                    # G_break = G_start + L_break
                    breaks_by_doc[item["doc"]].append(item["w0"] + local)
            if progress:
                print(f"    generated {min(bs+batch_size, len(order))}/{len(order)} "
                      f"windows", flush=True)
    finally:
        tokenizer.padding_side = prev_side

    for di in range(len(docs)):
        results[di] = (dedupe(breaks_by_doc[di], C.MATCH_TOLERANCE_WORDS),
                       unloc_by_doc[di])
    return results


def predict_document(words, model, tokenizer, max_new_tokens, device):
    """Single-document convenience wrapper over the batched path."""
    return predict_documents([{"words": words}], model, tokenizer,
                             max_new_tokens, device)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model",
                    default=str(C.PREPARED_MODEL) if C.PREPARED_MODEL.exists()
                    else C.MODEL_NAME)
    ap.add_argument("--split", choices=["train", "val", "test"],
                    help="run on the annotated transcripts of this split")
    ap.add_argument("--input", help="jsonl with record_id + body (overrides --split)")
    ap.add_argument("--out", default=None)
    # 256 is ample: an 18-break window needs ~380 tokens of anchors, and the
    # median target is well under 100. Generation stops at EOS anyway, so this
    # only bounds a rambling model — which is exactly the case worth bounding.
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--gen_batch_size", type=int, default=8,
                    help="windows generated concurrently; raise until OOM")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.input:
        docs = load_jsonl(args.input)
        out_path = args.out or (C.BUILD_DIR / "predictions_custom.jsonl")
    else:
        if not args.split:
            ap.error("pass --split or --input")
        rows = load_jsonl(C.BUILD_DIR / f"dataset_{args.split}.jsonl")
        wanted = {r["record_id"] for r in rows}
        docs = [d for d in load_jsonl(C.TRANSCRIPTS) if d["record_id"] in wanted]
        out_path = args.out or (C.BUILD_DIR / f"predictions_{args.split}.jsonl")

    if args.limit:
        docs = docs[:args.limit]
    print(f"{len(docs)} documents -> {out_path}")

    apply_saved_geometry(args.adapter)
    model, tokenizer = load_model(args.adapter, args.base_model)
    device = next(model.parameters()).device

    # Chunked so a huge corpus does not build one giant window pool in memory,
    # while still batching generation across documents inside each chunk.
    CHUNK = 64
    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, len(docs), CHUNK):
            chunk = docs[start:start + CHUNK]
            payload = [{"words": d["body"].split()} for d in chunk]
            preds = predict_documents(payload, model, tokenizer,
                                      args.max_new_tokens, device,
                                      batch_size=args.gen_batch_size,
                                      progress=args.progress)
            for d, p, (breaks, unloc) in zip(chunk, payload, preds):
                f.write(json.dumps({
                    "record_id": d["record_id"],
                    "outlet": d.get("outlet"),
                    "date": d.get("date"),
                    "word_count": len(p["words"]),
                    "pred_breaks": breaks,
                    "n_unlocatable_anchors": unloc,
                }) + "\n")
            print(f"  {min(start+CHUNK, len(docs))}/{len(docs)} documents",
                  flush=True)

    print(f"wrote {out_path}\nNext: python evaluate.py --predictions {out_path} "
          f"--split {args.split or 'test'}")


if __name__ == "__main__":
    main()
