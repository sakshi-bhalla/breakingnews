# Limitations

Eleven caveats, reproduced verbatim from the project handoff. The model is a
research instrument headed for a political-analysis paper in a methods journal;
every one of these is something a referee would raise, and none of them should
be discovered downstream.

This document exists so that nobody cites 0.680 as an accuracy figure.

---

**C1. Precision is a lower bound, not an estimate.**
A large share of scored false positives are *real* topic changes that the
annotator chose to group into a single thematic block. This is a disagreement
about granularity, not a model error. **216 such cases are still awaiting human
adjudication.** True precision is somewhere above 0.573 and nobody yet knows
where. Never present precision as settled, and never compute a derived
statistic that treats FP as clean error.

**C2. The test split is 20 transcripts and 64 gold breaks. It is too small to
carry a headline.**
At n=64, the binomial standard error on recall alone is ≈0.05, putting test F1
0.680 at roughly ±0.07 before any other source of variation. The val→test rise
(0.596 → 0.680) is **not** evidence the model generalises well; test recall is
0.797 against val's 0.621, which says the test transcripts have easier breaks.
Do not compare across splits. Do not report the higher number because it is
higher.

**C3. Every number is single-seed.**
One seed (42), one run, throughout — including the incumbent. There is no
variance estimate for any published result. A 3-seed replication is planned and
has not been run.

**C4. Differences below ~0.03 F1 are not distinguishable.**
Bootstrap on the 117-doc validation set: SD of absolute F1 ≈ **0.025**, SD of
the *paired* difference between two models ≈ **0.029**. Any comparison inside
that band is a statement about which transcripts landed in the split. This is
why the adoption rule for new configs is set at 0.029.

**C5. τ = 0.010 was itself selected on 117 documents.**
The threshold carries the same selection noise as anything else fitted on that
split. It is applied unchanged to test, which is the correct protocol and does
*not* make τ noise-free. If you expose τ, expose it as tunable.

**C6. Windowing coverage is near-total, but the two published figures measure
different things — don't conflate them.**
`docs/MODELS.md` reports **97.1% coverage** for `w3072_s1536_g200`; that is a
*training-row* figure — the share of gold breaks that yield a training example.
Measured separately at *inference* time on the val split, **0 of 322 gold breaks
had zero eligible windows**, so windowing imposes no meaningful recall ceiling
on that split. Quote whichever you mean and say which it is.

**C7. The deployed geometry has no high-precision regime.**
`V2_w3072` — the geometry V4_hard inherits — **cannot exceed precision 0.564 at
any threshold**; its confidences are saturated. `V2_w2048` reaches precision
**0.810** at identical peak F1, because its confidences are better spread. If a
downstream use is sensitive to false boundaries, raising τ on this model will
not deliver, and the fix is a different geometry, not a different threshold.

**C8. Domain is narrow.**
US broadcast-news transcripts from a specific set of outlets, hand-annotated by
one annotator. Behaviour on print, on other outlets, on other languages, or on
non-news speech is **unmeasured**. Not "probably fine" — unmeasured.

**C9. One annotator, no inter-annotator agreement.**
There is no second pass and no κ. The ceiling on measurable accuracy is unknown
because the reliability of the labels themselves is unknown. This compounds C1:
the granularity disagreements in C1 are disagreements with a single person's
judgement.

**C10. Boundaries only.**
The model locates seams. It does not identify topic, story type, or segment
boundaries' confidence in any calibrated sense. τ is a decision threshold, not a
probability you should report as one without calibration work.

**C11. Hard-example gain is one measurement.**
V4_hard's advantage over the un-mined baseline (test 0.680 vs 0.620) is a single
paired comparison on the 64-break test split — see C2 and C4. The direction is
plausible and the magnitude is not established.

---

## Two things this package adds to the list

**L1. No calibration layer.** `P("1")` is a raw thresholded logit. There is no
temperature scaling, no prior adjustment, and no conformal set. Anything the
package returns or prints that looks like a confidence is not one (C10).

**L1b. Predicted offsets are reproducible in aggregate, not bit-exact.**
Batched generation left-pads every prompt to the longest in its batch, and in
bf16 that padding perturbs the numerics enough to occasionally change a
generated anchor — which moves the word offset it localises to. Measured on one
8,909-word transcript: five of six boundaries were identical at every batch
size, one moved **+7 words** at batch 1, and a different one moved **−29 words**
at batch 4, while batches 8 and 16 reproduced the reference exactly. Batch
*composition* matters as well as batch size, so pooling a different set of
documents together can shift a boundary without changing any setting.

Measured on the full 20-document test split against the reference predictions:
**16 of 20 documents were offset-identical**, 4 boundaries shifted by at most 13
words, and one extra boundary was emitted (87 against 86). Recall was
*identical* at ±25, ±50 and ±100 — the same true boundaries are found — so the
whole difference is one additional false positive, worth **ΔF1 −0.005**. That is
roughly six times smaller than C4's 0.029 band, i.e. not a distinguishable
difference.

A plausible mechanism for the extra boundary is that a 13-word shift stopped two
near-duplicate predictions from merging inside the 25-word dedupe radius; that
has not been confirmed.

If you need byte-identical output across runs, fix `gen_batch_size` *and* the
document ordering. This is a property of batched bf16 inference generally, not
of this adapter.

**L2. The inference guard band is wider than the training one, deliberately.**
Training clipped a 200-token guard at each window edge (6.5% of a 3072-token
window); inference discards 10% of each window's *words*. Every published number
was produced at 10%, so 10% is the default. `Geometry.guard_fraction` exposes it
because narrowing it to match training is a plausible free-recall experiment —
one that has not been run.

## Where these numbers come from

Validation is 117 transcripts and 322 gold breaks; test is 20 transcripts and 64
gold breaks, verified by hand offset-by-offset and held out from every training
run. Trained on 998 annotated transcripts containing 2,829 boundaries, sampled
from US TV news broadcasts 1992–2020 (CNN, FOX, MSNBC, ABC, CBS).

The training transcripts are licensed material and are not distributed. The
annotations — word offsets carrying no text — are available for review on
request.
