"""
Stage 0.5 — build the prepared base model ONCE and save it.

  python prepare_base.py            # writes config.PREPARED_MODEL
  python prepare_base.py --verify   # check an existing prepared model

Registering <|STORY_BREAK|> (architecture.md §1) means resizing the embedding
matrix, and `mean_resizing=True` initialises the new row from the mean and
covariance of the existing 128,256 x 4,096 embeddings.  That covariance is
expensive: MEASURED AT 49.9 SECONDS, which is more than the model load (9.2s)
and the dataset tokenization (8-11s) combined.

Every HPO trial was paying it to produce a bit-identical result.  Over a 30-trial
search that is ~25 minutes of the budget spent recomputing one constant.

So: do it once here, save the model with the token already registered and the
embeddings already resized, and let every downstream run load that instead.
The saved model is numerically identical to what each trial was rebuilding.

Costs ~16GB of disk. Loading it is the same ~9s as loading the original.
"""
import argparse
import shutil

import torch

import config as C


def build(force=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = C.PREPARED_MODEL
    if out.exists() and not force:
        raise SystemExit(f"{out} already exists — pass --force to rebuild.")
    if out.exists():
        shutil.rmtree(out)

    print(f"base   : {C.MODEL_NAME}")
    print(f"target : {out}")

    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME)
    before = len(tokenizer)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [C.STORY_BREAK_TOKEN]})
    if tokenizer.pad_token is None:
        # Llama-3.1 ships without a pad token; generation and the collator both
        # need one, and eos is the conventional choice since padded positions
        # are masked out of both attention and loss anyway.
        tokenizer.pad_token = tokenizer.eos_token
    break_id = tokenizer.convert_tokens_to_ids(C.STORY_BREAK_TOKEN)

    pieces = tokenizer.tokenize(C.STORY_BREAK_TOKEN)
    print(f"\n{C.STORY_BREAK_TOKEN} -> id {break_id}, tokenizes to {pieces}")
    assert len(pieces) == 1, "special token did not register atomically"
    print(f"vocab {before} -> {len(tokenizer)} (added {added})")

    print("\nloading base weights...")
    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa")

    if added:
        print("resizing embeddings with mean_resizing (the ~50s step, once)...")
        model.resize_token_embeddings(len(tokenizer), mean_resizing=True)

    # mean_resizing draws the new row from N(mean, cov) of the existing rows.
    # But the MEAN embedding row is near zero — 128k rows pointing in every
    # direction largely cancel — so the sample lands with a much smaller norm
    # than a real token (measured: 0.085 vs 0.672 for embed_tokens, 0.226 vs
    # 0.906 for lm_head). A short-norm lm_head row produces a systematically
    # suppressed logit, so the token loses the argmax to ordinary punctuation no
    # matter what the hidden state says. Rescaling to the average row norm gives
    # it a fair starting point; direction (the part mean_resizing actually
    # informs) is preserved.
    with torch.no_grad():
        for name, mat in (("embed_tokens", model.get_input_embeddings().weight),
                          ("lm_head", model.get_output_embeddings().weight)):
            target = mat.norm(dim=1).mean()
            before = mat[break_id].norm()
            if before > 0:
                mat[break_id] *= (target / before)
            print(f"{name:<13} row[{break_id}] norm {before.item():.4f} -> "
                  f"{mat[break_id].norm().item():.4f}  (mean {target.item():.4f})")

    emb = model.get_input_embeddings().weight
    print(f"embedding matrix: {tuple(emb.shape)}  dtype {emb.dtype}")
    print(f"tied embeddings: "
          f"{emb.data_ptr() == model.get_output_embeddings().weight.data_ptr()}"
          f"   (False on Llama-3.1 -> lm_head must be trained separately)")

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    size = sum(f.stat().st_size for f in out.rglob("*")) / 1e9
    print(f"\nwrote {out}  ({size:.1f} GB)")
    print("Downstream runs load this and skip the resize entirely.")


def verify():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = C.PREPARED_MODEL
    if not out.exists():
        raise SystemExit(f"{out} missing — run: python prepare_base.py")

    tok = AutoTokenizer.from_pretrained(out)
    newly_added = tok.add_special_tokens(
        {"additional_special_tokens": [C.STORY_BREAK_TOKEN]})
    bid = tok.convert_tokens_to_ids(C.STORY_BREAK_TOKEN)
    print(f"tokenizer      vocab {len(tok)}  {C.STORY_BREAK_TOKEN} -> {bid}")
    print(f"  registers atomically : {tok.tokenize(C.STORY_BREAK_TOKEN)}")
    print(f"  needs re-adding      : {bool(newly_added)}  (must be False)")
    print(f"  pad token            : {tok.pad_token}")

    import time
    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    load_s = time.time() - t
    emb = model.get_input_embeddings().weight
    print(f"model          {tuple(emb.shape)} loaded in {load_s:.1f}s")
    ok = emb.shape[0] == len(tok) and not newly_added
    print(f"\n{'READY' if ok else 'NOT READY'} — embedding rows "
          f"{emb.shape[0]} vs vocab {len(tok)}")
    print(f"saves ~50s per run vs resizing from {C.MODEL_NAME}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    verify() if args.verify else build(force=args.force)
