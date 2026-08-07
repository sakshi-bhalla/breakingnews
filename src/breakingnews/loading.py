"""Resolving adapters and assembling the model.

No pattern for this existed in the reference packages -- they bundle their
weights. Ours cannot be bundled: the adapter is 168 MB against PyPI's 100 MB
per-file limit, so it lives on the Hub and is fetched once into the standard
cache.

The delicate part is the added `<|STORY_BREAK|>` token. Training resized the
embedding matrix and learned that row; the adapter stores it as a *delta*::

    base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta
    base_model.model.lm_head.token_adapter.trainable_tokens_delta

so the correct final row is `base_row + delta` and the base row has to be the
one training saw. See `install_break_token` for how that is guaranteed.
"""

from __future__ import annotations

import json
import logging
import struct
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import BREAK_TOKEN_ID, DEFAULT_BASE_MODEL, ROWS_FILENAME, PromptSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

_ROW_KEYS = ("embed_tokens", "lm_head")


def resolve_adapter(adapter: str | Path, *, revision: str | None = None) -> Path:
    """Resolve an adapter reference to a local directory.

    Args:
        adapter: Either a path to a local adapter directory or a Hugging Face
            Hub repo id such as `"sakshib3/breakingnews"`.
        revision: Hub branch, tag or commit sha to pin. **Leaving this None
            fetches `main`, which moves.** A published result should name a
            revision: without one, re-running the same package version a year
            later can load different weights and no error will say so. The
            resolved revision is logged either way.

    Returns:
        A local directory containing the adapter.

    Raises:
        FileNotFoundError: If a path-like reference does not exist and does not
            look like a Hub repo id.
    """
    path = Path(adapter)
    if path.is_dir():
        return path
    if path.exists() or path.is_absolute() or str(adapter).startswith("."):
        msg = (
            f"adapter directory not found: {adapter}. A Hub repo id is treated "
            "as one only if it is not path-like, so it must not be absolute or "
            "start with '.' -- e.g. 'sakshib3/breakingnews'."
        )
        raise FileNotFoundError(msg)

    from huggingface_hub import snapshot_download

    logger.info(
        "fetching adapter %s from the Hub at revision %s",
        adapter,
        revision or "main (unpinned)",
    )
    return Path(snapshot_download(repo_id=str(adapter), revision=revision))


def _safetensors_keys(path: Path) -> list[str]:
    """List the tensor names in a safetensors file without loading it.

    Deliberately stdlib-only. `verify_adapter` is advertised as running in the
    base install, so it cannot import safetensors (let alone torch) merely to
    read a header. The format is a little-endian uint64 header length followed
    by that many bytes of JSON.

    Args:
        path: A `.safetensors` file.

    Returns:
        Tensor names, excluding the `__metadata__` key.
    """
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return [k for k in header if k != "__metadata__"]


def verify_adapter(adapter_dir: str | Path) -> list[str]:
    """Check an adapter directory for the failure modes that fail silently.

    Every problem listed here produces wrong output rather than an exception,
    which is why they are checked rather than documented. Used by
    `breakingnews check-adapter` and by the artefact tests.

    Args:
        adapter_dir: A resolved local adapter directory.

    Returns:
        Human-readable problem descriptions; empty when the adapter is sound.
    """
    d = Path(adapter_dir)
    problems: list[str] = []

    cfg_path = d / "adapter_config.json"
    if not cfg_path.exists():
        return [f"missing adapter_config.json in {d}"]
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    if not (d / "segmentation_config.json").exists():
        problems.append(
            "missing segmentation_config.json -- window geometry is unknown and "
            "running at default geometry silently costs accuracy"
        )

    # Flagged even when the path resolves on this machine: it is a machine-local
    # path, so publishing the adapter with it set breaks for everyone else.
    base = str(cfg.get("base_model_name_or_path", ""))
    if base.startswith("/"):
        here = (
            "exists here but is machine-local"
            if Path(base).is_dir()
            else "does not exist here"
        )
        problems.append(
            f"base_model_name_or_path is an absolute path ({here}): {base}. "
            "Rewrite it to a Hub id before publishing."
        )

    # Llama-3.1 has tie_word_embeddings=False, so embed_tokens and lm_head are
    # separate matrices. Training only embed_tokens leaves the model unable to
    # EMIT the new token -- a run that scores F1 exactly 0 while eval_loss falls
    # normally. Both deltas must be present.
    tti = cfg.get("trainable_token_indices") or {}
    problems.extend(
        f"adapter_config.trainable_token_indices lacks {key!r}"
        for key in _ROW_KEYS
        if key not in tti
    )

    weights = d / "adapter_model.safetensors"
    if weights.exists():
        names = _safetensors_keys(weights)
        problems.extend(
            f"adapter weights carry no trainable-token delta for {key}"
            for key in _ROW_KEYS
            if not any(
                f"{key}.token_adapter.trainable_tokens_delta" in n for n in names
            )
        )
    else:
        problems.append("missing adapter_model.safetensors")

    return problems


def load_break_token_rows(adapter_dir: Path) -> dict[str, torch.Tensor] | None:
    """Load the absolute `<|STORY_BREAK|>` embedding rows, if shipped.

    Args:
        adapter_dir: A resolved local adapter directory.

    Returns:
        Mapping of `"embed_tokens"` / `"lm_head"` to 1-D tensors, or None when
        the artefact is absent.

    Raises:
        ValueError: If the file exists but lacks either row. A partial artefact
            is worse than none: it would seat one matrix correctly and leave the
            other at its redrawn value.
    """
    path = adapter_dir / ROWS_FILENAME
    if not path.exists():
        return None
    from safetensors.torch import load_file

    rows = load_file(str(path))
    missing = [k for k in _ROW_KEYS if k not in rows]
    if missing:
        msg = f"{path} is missing rows for {missing}"
        raise ValueError(msg)
    return {k: rows[k].reshape(-1) for k in _ROW_KEYS}


def export_break_token_rows(prepared_base: str | Path, out_path: str | Path) -> Path:
    """Extract the two absolute embedding rows from a prepared base model.

    Produces the 16 KB artefact that makes an adapter exactly reproducible.
    Run once against the base the adapter was trained on, then publish the
    result alongside the adapter.

    Args:
        prepared_base: Directory of the base model with `<|STORY_BREAK|>`
            already registered and its embeddings already resized.
        out_path: Destination `.safetensors` file.

    Returns:
        The path written.

    Raises:
        ValueError: If either embedding matrix is missing from the base, which
            means it is not a prepared base and the rows would be meaningless.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(prepared_base)
    shards = sorted(src.glob("*.safetensors"))
    wanted = {"model.embed_tokens.weight": "embed_tokens", "lm_head.weight": "lm_head"}
    rows: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for tensor_name, out_name in wanted.items():
                if tensor_name in f.keys():  # noqa: SIM118 - safetensors handle
                    rows[out_name] = f.get_tensor(tensor_name)[BREAK_TOKEN_ID].clone()
    missing = [k for k in _ROW_KEYS if k not in rows]
    if missing:
        msg = f"could not find rows for {missing} under {src}"
        raise ValueError(msg)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(rows, str(out))
    return out


def install_break_token(
    model: Any,
    tokenizer: Any,
    rows: dict[str, torch.Tensor] | None,
) -> None:
    """Resize the embedding matrix and seat the `<|STORY_BREAK|>` row.

    Two paths, and the difference matters more than it looks.

    With `rows`, the matrix is resized with `mean_resizing=False` and the two
    saved rows are written in directly. This is both exact and fast -- it skips
    the ~50 s covariance that `mean_resizing=True` computes over the whole
    128k x 4096 matrix on every single load.

    Without `rows`, transformers redraws the row. Measured, that draw is
    near-deterministic: the covariance is scaled by `epsilon = 1e-9` before
    sampling, so the sample is essentially the mean embedding vector
    (cos = 0.9999999), two independent draws agree to a relative 3.5e-4, and
    the reconstructed final row lands within a relative 1.7e-3 of the trained
    one -- at or below bf16 quantisation noise. So this path works today. It
    depends on an undocumented constant inside transformers, and removing that
    epsilon would be a defensible upstream change, at which point the row
    becomes a free draw and the model silently degrades. Hence the warning.

    Args:
        model: A causal LM whose vocabulary is one token short.
        tokenizer: Tokenizer with `<|STORY_BREAK|>` already registered.
        rows: Absolute rows from `load_break_token_rows`, or None.
    """
    import torch

    if model.get_input_embeddings().weight.shape[0] >= len(tokenizer):
        return  # already prepared, e.g. a locally saved prepared base

    if rows is None:
        warnings.warn(
            f"{ROWS_FILENAME} not found beside the adapter. Falling back to "
            "transformers' mean_resizing, which reconstructs the trained "
            "<|STORY_BREAK|> row to a relative ~2e-3 and costs ~50 s per load. "
            "Publish the 16 KB rows artefact to make loading exact and fast.",
            RuntimeWarning,
            stacklevel=2,
        )
        model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
        return

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    with torch.no_grad():
        for name, mat in (
            ("embed_tokens", model.get_input_embeddings().weight),
            ("lm_head", model.get_output_embeddings().weight),
        ):
            row = rows[name].to(dtype=mat.dtype, device=mat.device)
            mat[BREAK_TOKEN_ID].copy_(row)


def load_model(
    adapter_dir: Path,
    *,
    base_model: str | Path | None = None,
    dtype: str = "bfloat16",
    device_map: str | dict[str, Any] = "auto",
    attn_implementation: str = "sdpa",
) -> tuple[Any, Any]:
    """Assemble the tokenizer, base model and adapter.

    Args:
        adapter_dir: A resolved local adapter directory.
        base_model: Base to load the adapter onto. Defaults to the adapter's own
            `base_model_name_or_path` when that path exists locally -- a
            prepared base already carries the resized matrix -- and otherwise to
            the ungated Llama-3.1-8B-Instruct mirror, which is then prepared
            in memory.
        dtype: Torch dtype name for the base weights.
        device_map: Passed to `from_pretrained`.
        attn_implementation: Passed to `from_pretrained`.

    Returns:
        A `(model, tokenizer)` pair, with the model in eval mode.

    Raises:
        RuntimeError: If the model did not fit on the GPU and `device_map`
            silently offloaded part of it. Detected at load rather than left to
            surface as a meta-tensor error during generation.
        ValueError: If the adapter's tokenizer does not resolve
            `<|STORY_BREAK|>` to the expected id, which means it is not the
            tokenizer the adapter was trained with. Loading anyway would
            produce wrong output rather than an error.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "left"  # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    spec = PromptSpec()
    break_id = tokenizer.convert_tokens_to_ids(spec.story_break_token)
    if break_id != BREAK_TOKEN_ID:
        msg = (
            f"{spec.story_break_token} resolves to id {break_id}, expected "
            f"{BREAK_TOKEN_ID}. The tokenizer shipped with this adapter is not "
            "the one it was trained with."
        )
        raise ValueError(msg)

    # Resolved into its own non-optional name rather than reassigning the
    # parameter: the adapter's recorded base comes out of JSON as Any, so the
    # narrowing is invisible to a type checker and `None` could reach
    # from_pretrained, where it fails deep inside transformers.
    resolved_base: str | Path
    if base_model is not None:
        resolved_base = base_model
    else:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text("utf-8"))
        recorded = cfg.get("base_model_name_or_path")
        resolved_base = (
            str(recorded)
            if recorded and Path(str(recorded)).is_dir()
            else DEFAULT_BASE_MODEL
        )
        if resolved_base == DEFAULT_BASE_MODEL and recorded != DEFAULT_BASE_MODEL:
            logger.info(
                "adapter records base %r which is not present here; using %s",
                recorded,
                DEFAULT_BASE_MODEL,
            )

    model = AutoModelForCausalLM.from_pretrained(
        resolved_base,
        dtype=getattr(torch, dtype),
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    # `device_map="auto"` does not fail when the GPU is too full -- it offloads
    # layers to CPU or leaves them on the meta device, and the run continues
    # until generation dies with "Cannot copy out of meta tensor", which is not
    # an OOM and which the batch-size backoff cannot fix. Catching it here costs
    # a dict scan; missing it costs a model load and a confusing error.
    offloaded = {
        str(p.device.type)
        for p in model.parameters()
        if p.device.type in {"meta", "cpu", "disk"}
    }
    if offloaded and device_map == "auto":
        free = ""
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            free = (
                f" GPU has {free_b / 2**30:.1f} GiB free of {total_b / 2**30:.1f} GiB."
            )
        msg = (
            f"the model did not fit on the GPU: some parameters are on "
            f"{sorted(offloaded)}.{free} Running from here fails later with a "
            "meta-tensor error that looks like a model problem. Free the GPU, "
            "or pass device_map explicitly if you intend to offload."
        )
        raise RuntimeError(msg)

    install_break_token(model, tokenizer, load_break_token_rows(adapter_dir))
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer
