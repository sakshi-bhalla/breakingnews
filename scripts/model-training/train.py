"""
Stage 2 — LoRA fine-tuning for TV news story segmentation.

  python train.py --run_name seg_lora_v1

Deliberately plain: transformers.Trainer + peft, bf16, gradient checkpointing.
No TRL (its DataCollatorForCompletionOnlyLM no longer exists as of trl 1.9, and
SFTTrainer contributes nothing once the collator does its own masking), no
Unsloth (it pins transformers <5 and would downgrade the geoparser venv), and
no bitsandbytes (an 8B LoRA in bf16 needs ~25GB, which fits an A100-80GB with
room to spare — 4-bit would cost accuracy for no benefit).  See DECISIONS.md D3.

Masking: the prompt is masked to -100 and the whole target is trained on.  Under
the anchor format the target IS the boundary decision — roughly 20 tokens per
break, or the single token NONE — so architecture.md §4's per-category masking
table collapses to this.  See DECISIONS.md D1.
"""
import argparse
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import Trainer

import config as C

IGNORE_INDEX = -100


# ── Dataset ────────────────────────────────────────────────────────────────────

def render_prompt(row: Dict) -> str:
    return C.PROMPT_TEMPLATE.format(
        pre=C.ANCHOR_PRE_WORDS,
        post=C.ANCHOR_POST_WORDS,
        tok=C.STORY_BREAK_TOKEN,
        none=C.NO_BREAK_TARGET,
        input_text=row["input_text"],
    )


class SegmentationDataset(Dataset):
    """
    Pre-tokenizes once at construction so the collator does nothing but pad.
    Building input_ids as [BOS] + prompt + target + [EOS] with the label mask
    derived from the same concatenation removes any chance of the prompt-length
    off-by-one that arises from tokenizing the two halves under different
    add_special_tokens settings.
    """

    def __init__(self, path, tokenizer, max_len: int,
                 decision_weight: float = 1.0, break_weight: float = 1.0,
                 balance_cat_a_share: float = 0.0,
                 hard_spans=None, hard_weight: float = 1.0):
        self.examples = []
        self.skipped = []
        break_token_id = tokenizer.convert_tokens_to_ids(C.STORY_BREAK_TOKEN)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

        # CLASS BALANCE BY WEIGHT, NOT BY DELETION.
        #
        # Downsampling Cat C would fix the prior by throwing rows away; every
        # discarded window is annotation effort that never reaches the model.
        # Instead the decision token carries a per-class weight chosen so the
        # GRADIENT splits at the target ratio while all rows are still trained.
        #
        # Total decision mass is held at decision_weight * n_rows and merely
        # redistributed, so the loss scale stays comparable across settings.
        w_a = w_c = decision_weight
        if balance_cat_a_share > 0:
            n_a = sum(1 for r in rows if r["category"] == "true_transition")
            n_c = len(rows) - n_a
            if n_a and n_c:
                ra = balance_cat_a_share / n_a
                rc = (1.0 - balance_cat_a_share) / n_c
                scale = decision_weight * len(rows) / (n_a * ra + n_c * rc)
                w_a, w_c = ra * scale, rc * scale
                print(f"class-balanced decision weights -> target Cat A share "
                      f"{balance_cat_a_share:.2f}")
                print(f"  Cat A {n_a} rows x {w_a:.3f}   "
                      f"NONE {n_c} rows x {w_c:.3f}")
                print(f"  gradient share at the decision token: "
                      f"{100*n_a*w_a/(n_a*w_a+n_c*w_c):.1f}% Cat A "
                      f"(was {100*n_a/len(rows):.1f}%) — 0 rows dropped")

        # HARD-EXAMPLE MINING.
        #
        # hard_spans maps record_id -> [word offsets where the model erred on
        # THIS SPLIT]. A row whose window covers one of them gets its whole
        # target loss multiplied by hard_weight.
        #
        # Note what this does and does not do: the training data already carries
        # the correct label at every one of those offsets - the model saw the
        # window, saw no break marked, and fired anyway. Adding duplicate rows
        # would change nothing. Raising the weight is the only thing that shifts
        # the gradient toward the cases it actually gets wrong.
        hard_spans = hard_spans or {}
        n_hard = 0

        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        for row in rows:
            p_ids = tokenizer(render_prompt(row), add_special_tokens=False)["input_ids"]
            t_ids = tokenizer(row["target_text"], add_special_tokens=False)["input_ids"]

            input_ids = bos + p_ids + t_ids + eos
            if len(input_ids) > max_len:
                # Truncating the transcript would desync the anchors from the
                # text they quote, so an overlong row is dropped, not cut.
                self.skipped.append((row["record_id"], len(input_ids)))
                continue

            labels = ([IGNORE_INDEX] * (len(bos) + len(p_ids))) + t_ids + eos

            # Per-token loss weights.
            #
            # The break/no-break decision is made at ONE position: the first
            # generated token ("1" for a boundary list vs "NONE" for silence).
            # On a Cat A row that token is ~1/30th of the target loss; on a
            # NONE row it is the entire loss. So at the single position where
            # the decision actually happens, the gradient is dominated by
            # "say NONE" — which is the collapse mechanism, exactly located.
            #
            # Up-weighting that position (and the break token itself) restores
            # what architecture.md §4's masking table was reaching for, without
            # masking the quoted span the model needs in order to emit anchors
            # that can be located back to a word offset.
            w = [0.0] * (len(bos) + len(p_ids)) + [1.0] * (len(t_ids) + len(eos))
            first_target = len(bos) + len(p_ids)
            if first_target < len(w):
                w[first_target] = (w_a if row["category"] == "true_transition"
                                   else w_c)
            for i, t in enumerate(t_ids):
                if t == break_token_id:
                    w[first_target + i] = break_weight

            if hard_weight != 1.0:
                errs = hard_spans.get(row["record_id"], ())
                lo, hi = row["win_word_start"], row["win_word_end"]
                if any(lo <= o < hi for o in errs):
                    w = [x * hard_weight for x in w]
                    n_hard += 1

            self.examples.append({
                "input_ids": input_ids,
                "labels": labels,
                "weights": w,
                "category": row["category"],
            })

        if hard_weight != 1.0:
            print(f"hard-example weighting: {n_hard}/{len(self.examples)} rows "
                  f"({100*n_hard/max(1,len(self.examples)):.1f}%) contain a mined "
                  f"error, weighted x{hard_weight}")

        if self.skipped:
            longest = max(n for _, n in self.skipped)
            print(f"[WARN] {path.name}: dropped {len(self.skipped)} rows over "
                  f"max_len={max_len} (longest {longest}). Raise MAX_SEQ_LEN.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        n = max(len(f["input_ids"]) for f in features)

        def pad(seq, value):
            return seq + [value] * (n - len(seq))

        return {
            "input_ids": torch.tensor(
                [pad(f["input_ids"], self.pad_token_id) for f in features]),
            "labels": torch.tensor(
                [pad(f["labels"], IGNORE_INDEX) for f in features]),
            "attention_mask": torch.tensor(
                [[1] * len(f["input_ids"]) + [0] * (n - len(f["input_ids"]))
                 for f in features]),
            "weights": torch.tensor(
                [pad(f["weights"], 0.0) for f in features], dtype=torch.float),
        }


class WeightedTrainer(Trainer):
    """
    Trainer with per-token loss weights.

    HF's default path computes an unweighted mean cross-entropy internally, so
    the weights have to be applied by recomputing the loss here.  Normalising by
    the SUM OF WEIGHTS (not the token count) keeps the loss scale comparable to
    the unweighted run — otherwise raising decision_weight would inflate the
    reported loss and look like a regression.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights")
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits[..., :-1, :].contiguous()
        tgt = labels[..., 1:].contiguous()
        w = weights[..., 1:].contiguous()

        per_tok = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), tgt.view(-1),
            reduction="none", ignore_index=IGNORE_INDEX)

        w_flat = w.reshape(-1) * (tgt.reshape(-1) != IGNORE_INDEX).float()
        denom = w_flat.sum().clamp_min(1e-8)
        loss = (per_tok * w_flat).sum() / denom
        return (loss, outputs) if return_outputs else loss


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model_and_tokenizer(model_name: str, gradient_checkpointing: bool,
                              lora_r: int = None, lora_dropout: float = None,
                              alpha_ratio: float = 2.0):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Prefer the model prepared by prepare_base.py: <|STORY_BREAK|> already
    # registered and embeddings already resized, which skips a 49.9s
    # mean_resizing covariance that produces an identical result every run.
    if model_name == C.MODEL_NAME and C.PREPARED_MODEL.exists():
        model_name = str(C.PREPARED_MODEL)
        print(f"using prepared base: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # <|STORY_BREAK|> must stay one atomic token instead of fragmenting into
    # ['<','|','ST','ORY','_BREAK','|','>'] (architecture.md §1). Already true
    # for the prepared base; this is the fallback path.
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [C.STORY_BREAK_TOKEN]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="sdpa",
    )
    break_id = tokenizer.convert_tokens_to_ids(C.STORY_BREAK_TOKEN)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        # mean_resizing seeds the new row from the embedding mean rather than
        # noise, so the fresh token starts in-distribution.
        model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    print(f"{C.STORY_BREAK_TOKEN} -> id {break_id} "
          f"({tokenizer.tokenize(C.STORY_BREAK_TOKEN)})")

    if gradient_checkpointing:
        model.config.use_cache = False

    # alpha is set as a RATIO to r, not independently: LoRA's effective update
    # scales as (alpha/r), so holding the ratio fixed keeps the update magnitude
    # comparable as r varies. Sampling them independently would confound
    # capacity with step size.
    r = lora_r if lora_r is not None else C.LORA_R
    lora = LoraConfig(
        r=r,
        lora_alpha=int(r * alpha_ratio),
        lora_dropout=(lora_dropout if lora_dropout is not None else C.LORA_DROPOUT),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=C.LORA_TARGET_MODULES,
        # The <|STORY_BREAK|> embedding row has to train, or it stays at its
        # initialisation and the model can never emit the token.  The blunt way
        # is modules_to_save=["embed_tokens","lm_head"], but that makes the
        # whole 128,257 x 4,096 matrix trainable — ~1.05B extra parameters and
        # 10-17GB of optimizer state to learn ONE row, which defeats LoRA.
        # BOTH embed_tokens and lm_head: Llama-3.1 has tie_word_embeddings=
        # False, so a bare list would train only the input row and leave the
        # output row frozen — the model could never emit the token at all.
        trainable_token_indices={"embed_tokens": [break_id],
                                 "lm_head": [break_id]},
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.MODEL_NAME)
    ap.add_argument("--data_dir", default=None,
                    help="dataset dir (default build/); e.g. "
                         "build/variants/w2048_s1024_g200")
    ap.add_argument("--seed", type=int, default=C.SEED,
                    help="~1 in 3 runs collapse to always-NONE at small data "
                         "sizes, so the final model is trained at >1 seed and "
                         "the non-collapsed one kept")
    ap.add_argument("--train_fraction", type=float, default=1.0,
                    help="fraction of TRANSCRIPTS to train on (learning curve)")
    ap.add_argument("--run_name", default="seg_lora_v1")
    ap.add_argument("--max_seq_len", type=int, default=C.MAX_SEQ_LEN)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--lora_r", type=int, default=None,
                    help="LoRA rank. Untested: at r=16 the model carries 11,600 "
                         "trainable params per training row and shows a +0.147 "
                         "train/val gap, so capacity is a live suspect.")
    ap.add_argument("--lora_dropout", type=float, default=None)
    ap.add_argument("--alpha_ratio", type=float, default=2.0,
                    help="alpha = ratio * r; keeps the effective update scale "
                         "constant as r varies")
    ap.add_argument("--resume_from", default=None,
                    help="checkpoint to continue from, so a trial can be grown "
                         "one epoch at a time and a pruner can stop it early "
                         "without discarding the epochs already trained")
    ap.add_argument("--stop_after_epoch", type=float, default=None,
                    help="Halt after this many epochs while STILL building the "
                         "optimiser schedule for the full --epochs. Successive "
                         "halving needs this: with lr_scheduler_type='cosine', "
                         "training 1 epoch with --epochs 1 decays the rate to ~0, "
                         "so resuming into a 2-epoch schedule would jump it back "
                         "up - a warm restart, not a continuation, and the rung-2 "
                         "model would not be the model a plain 2-epoch run gives. "
                         "Keeping --epochs at the ladder's TOP and truncating here "
                         "puts every rung on one identical cosine curve.")
    ap.add_argument("--save_epochs", action="store_true",
                    help="save an adapter checkpoint after every epoch. Lets one "
                         "training run yield N models to compare, instead of N "
                         "runs - the cheapest way to test the epoch count, which "
                         "is the only untested hyperparameter with evidence "
                         "behind it (eval_loss rises between epoch 2 and 3).")
    ap.add_argument("--no_eval", action="store_true",
                    help="skip the epoch-end validation pass. It costs ~3.5 min "
                         "per epoch over the full val split and only feeds "
                         "load_best_model_at_end, which selects on eval_loss - a "
                         "metric that has moved OPPOSITE to task F1 every time it "
                         "was checked here. Pure overhead for screening runs "
                         "scored on best-threshold F1.")
    ap.add_argument("--decision_weight", type=float, default=1.0,
                    help="loss weight on the FIRST target token, where the "
                         "break/no-break decision is made. >1 counteracts the "
                         "NONE-majority gradient that drives collapse.")
    ap.add_argument("--break_weight", type=float, default=1.0,
                    help="loss weight on <|STORY_BREAK|> token positions")
    ap.add_argument("--hard_examples", default=None,
                    help="jsonl of mined errors (build/trainerr_*.jsonl). Rows "
                         "whose window covers one get up-weighted. MUST be mined "
                         "on the TRAINING split - val-mined errors would move val "
                         "transcripts into training and inflate every later score.")
    ap.add_argument("--hard_weight", type=float, default=3.0,
                    help="loss multiplier for rows containing a mined error")
    ap.add_argument("--balance_cat_a_share", type=float, default=0.0,
                    help="rebalance the decision-token gradient to this Cat A "
                         "share WITHOUT dropping rows (0 = off). 0.57 matches "
                         "architecture.md §2's 40:30 A:C ratio with Cat B absent.")
    args = ap.parse_args()


    from transformers import TrainingArguments

    out_dir = C.RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    grad_ckpt = not args.no_grad_ckpt
    model, tokenizer = build_model_and_tokenizer(
        args.model, grad_ckpt, lora_r=args.lora_r,
        lora_dropout=args.lora_dropout, alpha_ratio=args.alpha_ratio)

    data_dir = Path(args.data_dir) if args.data_dir else C.BUILD_DIR
    print(f"data: {data_dir}")
    hard_spans = None
    if args.hard_examples:
        hard_spans = {}
        for f in args.hard_examples.split(","):
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                hard_spans.setdefault(r["record_id"], []).append(r["word_offset"])
        print(f"mined errors: {sum(len(v) for v in hard_spans.values())} across "
              f"{len(hard_spans)} transcripts")

    train_ds = SegmentationDataset(
        data_dir / "dataset_train.jsonl", tokenizer, args.max_seq_len,
        decision_weight=args.decision_weight, break_weight=args.break_weight,
        balance_cat_a_share=args.balance_cat_a_share,
        hard_spans=hard_spans,
        hard_weight=(args.hard_weight if args.hard_examples else 1.0))
    # Val loss stays UNWEIGHTED so it remains comparable across runs with
    # different weighting settings.
    val_ds = SegmentationDataset(
        data_dir / "dataset_val.jsonl", tokenizer, args.max_seq_len)
    if args.decision_weight != 1.0 or args.break_weight != 1.0:
        print(f"loss weights: decision token x{args.decision_weight}, "
              f"{C.STORY_BREAK_TOKEN} x{args.break_weight}")

    if args.train_fraction < 1.0:
        # By TRANSCRIPT, not by row: overlapping windows from one transcript are
        # not independent observations, so dropping random rows would leave
        # correlated siblings behind and overstate the effective sample size.
        import random as _r
        rows = [json.loads(l) for l in open(data_dir / "dataset_train.jsonl")]
        ids = sorted({r["record_id"] for r in rows})
        rng = _r.Random(args.seed)
        rng.shuffle(ids)
        keep = set(ids[:max(1, int(len(ids) * args.train_fraction))])
        train_ds.examples = [e for e, r in zip(train_ds.examples, rows)
                             if r["record_id"] in keep]
        print(f"train_fraction {args.train_fraction}: {len(train_ds)} rows "
              f"from {len(keep)} transcripts")

    from collections import Counter
    print(f"\ntrain rows {len(train_ds)}  "
          f"{dict(Counter(e['category'] for e in train_ds.examples))}")
    print(f"val   rows {len(val_ds)}  "
          f"{dict(Counter(e['category'] for e in val_ds.examples))}")

    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
    print(f"effective batch {eff_batch}, ~{steps_per_epoch} optimizer steps/epoch\n")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        optim="adamw_torch",
        logging_steps=5,
        eval_strategy=("no" if args.no_eval else "epoch"),
        # save_epochs keeps per-epoch checkpoints WITHOUT paying for the
        # validation pass - the two were coupled only by HF's defaults.
        save_strategy=("epoch" if (args.save_epochs or not args.no_eval) else "no"),
        save_total_limit=(None if args.save_epochs else 2),
        load_best_model_at_end=(not args.no_eval),
        metric_for_best_model=(None if args.no_eval else "eval_loss"),
        greater_is_better=(None if args.no_eval else False),
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=(None if args.no_eval else val_ds),
        data_collator=PadCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    if args.stop_after_epoch is not None:
        from transformers import TrainerCallback

        class StopAfterEpoch(TrainerCallback):
            def on_epoch_end(self, a, state, control, **kw):
                if state.epoch >= args.stop_after_epoch - 1e-6:
                    print(f"\n[rung] reached epoch {state.epoch:.2f}, halting "
                          f"(schedule remains {args.epochs}-epoch cosine)")
                    control.should_training_stop = True
                return control

        trainer.add_callback(StopAfterEpoch())

    print("Starting training...")
    # resume_from lets a trial be trained ONE EPOCH AT A TIME without redoing
    # the epochs already completed. That is what makes early pruning pay: a
    # config killed after epoch 1 costs one epoch, not three.
    trainer.train(resume_from_checkpoint=args.resume_from)

    model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    # Window geometry must travel WITH the adapter. A model trained on
    # 2,048-token windows and then run with config.py's 4,096 default sees a
    # different slicing of the transcript than it ever trained on, and its
    # anchors stop lining up. infer.py reads this back.
    meta_src = data_dir / "meta.json"
    geom = json.load(open(meta_src)) if meta_src.exists() else {
        "window_tokens": C.WINDOW_TOKENS, "stride_tokens": C.STRIDE_TOKENS,
        "edge_guard_lo": C.EDGE_GUARD_LO, "max_seq_len": C.MAX_SEQ_LEN}
    with open(out_dir / "adapter" / "segmentation_config.json", "w") as f:
        json.dump({k: geom[k] for k in
                   ("window_tokens", "stride_tokens", "edge_guard_lo",
                    "max_seq_len") if k in geom}, f, indent=2)
    print(f"window geometry saved: {geom.get('window_tokens')}/"
          f"{geom.get('stride_tokens')} guard {geom.get('edge_guard_lo')}")
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nSaved adapter -> {out_dir / 'adapter'}")
    print("Next: python evaluate.py --adapter "
          f"{out_dir / 'adapter'} --split test")


if __name__ == "__main__":
    main()
