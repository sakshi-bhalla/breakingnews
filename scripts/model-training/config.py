"""
lora_training — shared configuration.

Task: given a ~4,096-token slice of a TV news transcript, predict where the
macro story boundaries fall.  Trained as a LoRA adapter on Llama-3.1-8B-Instruct.

Three design decisions here depart from `architecture.md`; each is recorded in
`DECISIONS.md` with its rationale.  In short:
  D1  target = break ANCHORS, not the regurgitated window
  D2  Cat A = windows over the annotated dumps, not synthetic stitched pairs
  D3  bf16 LoRA on one 80GB GPU, no 4-bit / no Unsloth / no TRL

Nothing in this pipeline drops data silently.  Every window the router rejects
is written to the manifest with a `reject_reason`, so the researcher can see
exactly what the edge-guard removed and change it.
"""
import os
from pathlib import Path

# Every path in this file is overridable by an environment variable, so running
# on a different machine or corpus does not require editing it.  Unset, the
# defaults reproduce the layout the published run used.
#
#   BREAKINGNEWS_DATA_DIR       where transcripts/annotations live
#   BREAKINGNEWS_BUILD_DIR      generated dataset + manifests
#   BREAKINGNEWS_RUNS_DIR       training outputs (adapters, logs)
#   BREAKINGNEWS_TRANSCRIPTS    transcripts JSONL (see ../../schemas/)
#   BREAKINGNEWS_ANNOTATIONS    annotations JSONL (see ../../schemas/)
#   BREAKINGNEWS_PREPARED_BASE  the ~15GB prepared base model
#   HF_HOME                     Hugging Face cache (standard variable)

BASE = Path(__file__).resolve().parent


def _path(var, default):
    """Read a path from the environment, falling back to a default."""
    return Path(os.environ[var]) if os.environ.get(var) else default


DATA_DIR = _path("BREAKINGNEWS_DATA_DIR", BASE / "data")
BUILD_DIR = _path("BREAKINGNEWS_BUILD_DIR", BASE / "build")
RUNS_DIR = _path("BREAKINGNEWS_RUNS_DIR", BASE / "runs")

# ---- Source files ----------------------------------------------------------
TRANSCRIPTS = _path("BREAKINGNEWS_TRANSCRIPTS", DATA_DIR / "sample_transcripts.jsonl")
# Full annotation pass: 998 transcripts, 2,829 breaks — 2.06x the original
# batch480+tests set, and it supersedes them (it contains all 464 batch480 ids
# and all 20 verified test ids, with identical break offsets for the latter).
ANNOTATIONS = [
    (_path("BREAKINGNEWS_ANNOTATIONS", DATA_DIR / "all_annotations.jsonl"), False),
]

# Held out by RECORD_ID rather than by file, because the verified transcripts
# now also live inside all_annotations.jsonl.  These 20 were checked
# offset-by-offset by hand, so they stay the test split — a test set is only
# worth having if its labels are the ones you trust most.
HELDOUT_ID_SOURCES = [
    DATA_DIR / "test_batch_annotations.jsonl",
    DATA_DIR / "test2_batch_annotations.jsonl",
]

# ---- Model -----------------------------------------------------------------
# Ungated mirror of meta-llama/Llama-3.1-8B-Instruct: same 4 bf16 shards, same
# LlamaForCausalLM config, no HF token required.  Swap to the meta-llama repo
# if you would rather have official provenance and supply a token.
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"

# Point HF_HOME at shared storage on a cluster so the 16GB download happens
# exactly once and is visible from both login and compute nodes.  Defaults to
# the standard Hugging Face cache.
HF_HOME = _path("HF_HOME", Path.home() / ".cache" / "huggingface")

# Base model with <|STORY_BREAK|> already registered and the embedding matrix
# already resized, built once by prepare_base.py.  The resize itself is cheap;
# `mean_resizing=True` is not — it takes a covariance over the whole 128k x 4096
# embedding matrix, MEASURED at 49.9s, and every training run was recomputing
# the identical result.  Loading this instead costs the same ~9s as the original.
PREPARED_MODEL = _path("BREAKINGNEWS_PREPARED_BASE", BASE / "base_prepared")

# ---- Windowing (architecture.md §3) ----------------------------------------
WINDOW_TOKENS = 4096
STRIDE_TOKENS = 3072
# Edge guard: a break closer than this to either end of the window has too
# little prior/future context for the model to confirm a macro shift.
EDGE_GUARD_LO = 400
EDGE_GUARD_HI = 3696

# When a break lands in a guard band the window cannot be emitted as-is: the
# target claims to list EVERY boundary in the slice, so omitting that break
# would teach the model a manufactured false negative.
#
# The spec's implied fallback is to discard the window, but that also throws
# away every OTHER break in it — guard bands are 19.5% of a window and dense
# transcripts carry up to 18 breaks, so one unlucky break kills its siblings.
# Measured cost: 281 windows lost, taking 34% of all gold breaks with them.
#
# Instead the window EDGE is pulled inward past the offending break, so the
# slice that remains is completely and honestly labelled. The clipped-out break
# is not lost — it lands mid-window in the neighbour, which is what the
# 1,024-token overlap is for. Coverage 66.3% -> 91.5% at the same guard width.
MIN_WINDOW_TOKENS = 1024      # a clipped window shorter than this is dropped

# A trailing end-anchored window is appended so the transcript tail gets full
# context, but only when it advances this far past the last grid window.  It is
# not a pure duplicate even when it barely moves — the small shift lifts tail
# breaks out of the guard band — so the threshold trades row duplication against
# coverage.  Measured (with clipping on):
#     0 -> 1231 rows / 91.5% covered / 407 repeated breaks
#   256 -> 1191 rows / 91.0% covered / 333 repeated breaks   <- chosen
#   768 -> 1111 rows / 88.1% covered / 182 repeated breaks
#  none -> 839 rows  / 70.5% covered /  17 repeated breaks
TAIL_WINDOW_MIN_ADVANCE = 256
# Extra tokens to cut past the offending break, so a clipped edge does not sit
# exactly on a story boundary. Costs coverage (91.5% -> 90.8% at 128) for a
# speculative benefit; 0 unless an ablation shows the artifact matters.
CLIP_JITTER_TOKENS = 0

# ---- Target format (D1) ----------------------------------------------------
STORY_BREAK_TOKEN = "<|STORY_BREAK|>"
# Words quoted on each side of a break in the target.  12 pre-context words are
# unique within a +-1600-word neighbourhood for 99.93% of the 1,375 annotated
# breaks (8 words: 99.49%), which is what makes a generated anchor localisable
# back to a word offset by string search.
ANCHOR_PRE_WORDS = 12
ANCHOR_POST_WORDS = 8
NO_BREAK_TARGET = "NONE"

PROMPT_TEMPLATE = (
    "You are segmenting a television news transcript into distinct stories.\n"
    "Below is a slice of one broadcast. Mark every point where the broadcast "
    "moves from one story to a genuinely different story - a new topic, a new "
    "event, different actors. Do not mark a change of speaker, correspondent, "
    "location, or sub-angle within a continuing story.\n\n"
    "For each boundary, quote the {pre} words immediately before it, then "
    "{tok}, then the {post} words immediately after. "
    "If there are no boundaries, answer {none}.\n\n"
    "Transcript:\n{input_text}\n\nBoundaries:\n"
)

# ---- Splits ----------------------------------------------------------------
# Grouped by record_id.  Windows overlap by 1,024 tokens, so splitting at the
# window level would leak the same text across train and eval.
SEED = 42
VAL_FRACTION = 0.12          # of the non-heldout (batch480) transcripts

# ---- Training (D3) ---------------------------------------------------------
# 4,096-token window + ~120-token prompt scaffold + up to ~700 tokens of anchors
# on a dense window.  Measured max over the built set is 4,842; 5,120 leaves
# headroom without a meaningful attention-cost increase.
MAX_SEQ_LEN = 5120
LORA_R = 16
LORA_ALPHA = 32              # 2*r; architecture.md said 16, see DECISIONS.md D4
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ---- Evaluation ------------------------------------------------------------
# A predicted break counts as a hit if it lands within this many words of a
# gold break.  Annotators placed offsets at the first word of the new story;
# +-N absorbs disagreement about whether the tease/handoff belongs to the old
# story or the new one.
MATCH_TOLERANCE_WORDS = 25
