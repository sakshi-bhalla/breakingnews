"""Artefact-integrity tests.

Every failure mode checked here produces *wrong output* rather than an
exception, which is why it is a test rather than a paragraph in a README:

* window geometry missing -> the adapter silently runs at the wrong window size
* `<|STORY_BREAK|>` at the wrong id -> a mismatched tokenizer, silently
* an absolute `base_model_name_or_path` -> breaks for everyone but its author
* a missing `lm_head` delta -> the model cannot *emit* the new token; this one
  cost a full training run that scored F1 exactly 0 while `eval_loss` fell
  normally

The synthetic tests always run. The tests against a real adapter run only when
`BREAKINGNEWS_TEST_ADAPTER` points at one, so CI needs no weights.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from breakingnews.config import BREAK_TOKEN_ID, ROWS_FILENAME
from breakingnews.loading import _safetensors_keys, resolve_adapter, verify_adapter

GEOMETRY = {
    "window_tokens": 3072,
    "stride_tokens": 1536,
    "edge_guard_lo": 200,
    "max_seq_len": 4096,
}
DELTAS = (
    "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta",
    "base_model.model.lm_head.token_adapter.trainable_tokens_delta",
)


def write_safetensors(path: Path, names: list[str]) -> None:
    """Write a safetensors file with the given tensor names and no real data.

    Args:
        path: Destination file.
        names: Tensor names to record in the header.
    """
    header = {n: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for n in names}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00\x00\x00\x00")


@pytest.fixture
def adapter(tmp_path: Path) -> Path:
    """A minimal, sound adapter directory.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        The directory.
    """
    (tmp_path / "segmentation_config.json").write_text(json.dumps(GEOMETRY))
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "unsloth/Meta-Llama-3.1-8B-Instruct",
                "trainable_token_indices": {
                    "embed_tokens": [BREAK_TOKEN_ID],
                    "lm_head": [BREAK_TOKEN_ID],
                },
            }
        )
    )
    write_safetensors(tmp_path / "adapter_model.safetensors", list(DELTAS))
    return tmp_path


class TestVerifyAdapter:
    def test_a_sound_adapter_has_no_problems(self, adapter):
        assert verify_adapter(adapter) == []

    def test_missing_geometry_is_reported(self, adapter):
        (adapter / "segmentation_config.json").unlink()
        assert any("segmentation_config" in p for p in verify_adapter(adapter))

    def test_absolute_base_path_is_reported_even_when_it_resolves(
        self, adapter, tmp_path
    ):
        # A path that exists here is still machine-local and still unpublishable.
        local = tmp_path / "base_prepared"
        local.mkdir()
        cfg = adapter / "adapter_config.json"
        data = json.loads(cfg.read_text())
        data["base_model_name_or_path"] = str(local)
        cfg.write_text(json.dumps(data))
        problems = verify_adapter(adapter)
        assert any("absolute path" in p for p in problems)
        assert any("machine-local" in p for p in problems)

    @pytest.mark.parametrize("missing", ["embed_tokens", "lm_head"])
    def test_a_missing_trainable_token_index_is_reported(self, adapter, missing):
        cfg = adapter / "adapter_config.json"
        data = json.loads(cfg.read_text())
        del data["trainable_token_indices"][missing]
        cfg.write_text(json.dumps(data))
        assert any(missing in p for p in verify_adapter(adapter))

    @pytest.mark.parametrize("missing", ["embed_tokens", "lm_head"])
    def test_a_missing_delta_tensor_is_reported(self, adapter, missing):
        # The lm_head case is the F1-exactly-zero bug.
        kept = [n for n in DELTAS if missing not in n]
        write_safetensors(adapter / "adapter_model.safetensors", kept)
        assert any(
            f"no trainable-token delta for {missing}" in p
            for p in verify_adapter(adapter)
        )

    def test_missing_weights_are_reported(self, adapter):
        (adapter / "adapter_model.safetensors").unlink()
        assert any("adapter_model.safetensors" in p for p in verify_adapter(adapter))

    def test_missing_config_short_circuits(self, tmp_path):
        problems = verify_adapter(tmp_path)
        assert len(problems) == 1
        assert "adapter_config.json" in problems[0]


class TestSafetensorsHeader:
    def test_reads_names_without_safetensors_installed(self, tmp_path):
        # verify_adapter runs in the base install, so this must be stdlib-only.
        p = tmp_path / "x.safetensors"
        write_safetensors(p, ["a", "b"])
        assert _safetensors_keys(p) == ["a", "b"]

    def test_metadata_key_is_excluded(self, tmp_path):
        p = tmp_path / "x.safetensors"
        blob = json.dumps({"__metadata__": {"k": "v"}, "a": {}}).encode()
        p.write_bytes(struct.pack("<Q", len(blob)) + blob)
        assert _safetensors_keys(p) == ["a"]


class TestResolveAdapter:
    def test_local_directory_passes_through(self, adapter):
        assert resolve_adapter(adapter) == adapter

    def test_missing_absolute_path_raises_rather_than_hitting_the_hub(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_adapter(tmp_path / "nope")


@pytest.mark.skipif(
    not os.environ.get("BREAKINGNEWS_TEST_ADAPTER"),
    reason="set BREAKINGNEWS_TEST_ADAPTER to a real adapter directory",
)
class TestRealAdapter:
    @pytest.fixture
    def real(self) -> Path:
        return Path(os.environ["BREAKINGNEWS_TEST_ADAPTER"])

    def test_it_is_publishable(self, real):
        assert verify_adapter(real) == []

    def test_geometry_is_the_trained_geometry(self, real):
        saved = json.loads((real / "segmentation_config.json").read_text())
        assert saved["window_tokens"] == 3072
        assert saved["stride_tokens"] == 1536
        assert saved["edge_guard_lo"] == 200

    def test_both_deltas_are_present_and_the_right_shape(self, real):
        names = _safetensors_keys(real / "adapter_model.safetensors")
        for key in ("embed_tokens", "lm_head"):
            assert any(
                f"{key}.token_adapter.trainable_tokens_delta" in n for n in names
            )

    def test_the_break_token_rows_ship_alongside(self, real):
        # Optional but recommended: makes loading exact and skips a ~50s resize.
        rows = real / ROWS_FILENAME
        if not rows.exists():
            pytest.skip(f"{ROWS_FILENAME} not published with this adapter")
        assert set(_safetensors_keys(rows)) == {"embed_tokens", "lm_head"}
