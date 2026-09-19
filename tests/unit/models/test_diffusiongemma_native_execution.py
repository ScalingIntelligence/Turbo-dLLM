from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dllm_parallel.core.models.backbones.nemotron.model import (
    DiffusionGemmaNativeOutput,
    NemotronLabsDiffusionPackedBlockDiffusionModel,
    _DiffusionGemmaCleanStream,
    _soft_embedding_scale,
)


class _RecordingSelfConditioning(torch.nn.Module):
    def __init__(self, calls: list[tuple[str, bool, bool]]) -> None:
        super().__init__()
        self.calls = calls

    def forward(
        self,
        inputs: torch.Tensor,
        signal: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append(
            (
                "self_conditioning",
                torch.is_grad_enabled(),
                bool(torch.count_nonzero(signal)),
            )
        )
        return inputs + signal


class _NativeExecutionHarness(NemotronLabsDiffusionPackedBlockDiffusionModel):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.runtime = SimpleNamespace(
            enabled=True,
            kv_backend="ring",
            tensor_parallel_size=1,
            context_attention_size=1,
            block_parallel_size=1,
        )
        self.max_seq_len = 4
        self.seq_len = 4
        self.block_size = 2
        self.sequence_parallel = False
        self.encoder_causal_attention = True
        self.activation_checkpointing = True
        self.activation_checkpointing_scope = "full"
        self.calls: list[tuple[str, bool, bool]] = []
        self.decode_batch_sizes: list[int] = []
        self.decode_layout_metadata: list[
            tuple[torch.Tensor | None, torch.Tensor | None]
        ] = []
        self.last_self_conditioning_signal: torch.Tensor | None = None
        self.self_conditioning = _RecordingSelfConditioning(self.calls)
        self.embedding_scale = None
        self.output_multiplier = None
        self.final_logit_softcap = None
        self.output_head = torch.nn.Linear(3, 7, bias=False)
        self.embed_tokens = torch.nn.Embedding(7, 3)

    def _activate_sequence_length(self, sequence_length: int) -> None:
        self.seq_len = int(sequence_length)

    def _native_stream_positions(
        self,
        device: torch.device,
        *,
        decoder_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        length = 4 if decoder_length is None else int(decoder_length)
        positions = torch.arange(length, device=device)
        return positions, positions, False

    def _native_encode_clean(
        self,
        clean_input_ids: torch.Tensor,
        clean_positions: torch.Tensor,
        *,
        pure_context: bool = False,
    ) -> _DiffusionGemmaCleanStream:
        batch_size = int(clean_input_ids.shape[0])
        del pure_context
        hidden = torch.arange(
            batch_size * 12,
            dtype=torch.float32,
        ).reshape(batch_size, 4, 3)
        hidden.requires_grad_(True)
        key = (hidden * 2.0).reshape(batch_size, 4, 1, 3)
        value = (hidden * 3.0).reshape(batch_size, 4, 1, 3)
        self.calls.append(("clean", torch.is_grad_enabled(), key.requires_grad))
        return _DiffusionGemmaCleanStream(
            hidden=hidden,
            positions=clean_positions,
            layer_key_values=((key, value),),
        )

    def _native_embed_active(
        self,
        noisy_input_ids: torch.Tensor,
        active_positions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones(
            (
                int(noisy_input_ids.shape[0]),
                int(active_positions.numel()),
                3,
            ),
            requires_grad=True,
        )

    def _native_decode_active(
        self,
        active_hidden: torch.Tensor,
        active_positions: torch.Tensor,
        clean_stream: _DiffusionGemmaCleanStream,
        *,
        decoder_position_ids: torch.Tensor | None = None,
        decoder_block_ids: torch.Tensor | None = None,
        response_start: int = 0,
        pure_context: bool,
    ) -> torch.Tensor:
        self.decode_layout_metadata.append((decoder_position_ids, decoder_block_ids))
        del (
            active_positions,
            decoder_position_ids,
            decoder_block_ids,
            response_start,
            pure_context,
        )
        key = clean_stream.layer_key_values[0][0]
        assert int(active_hidden.shape[0]) == int(key.shape[0])
        self.decode_batch_sizes.append(int(active_hidden.shape[0]))
        self.calls.append(("decode", torch.is_grad_enabled(), key.requires_grad))
        return active_hidden + key[:, : active_hidden.shape[1], 0]

    def _native_apply_self_conditioning(
        self,
        active_hidden: torch.Tensor,
        first_pass_hidden: torch.Tensor,
        self_conditioning_mask: torch.Tensor,
        *,
        row_chunk_size: int,
        vocab_chunk_size: int,
    ) -> torch.Tensor:
        del row_chunk_size, vocab_chunk_size
        signal = first_pass_hidden * self_conditioning_mask[:, None, None]
        self.last_self_conditioning_signal = signal.detach().clone()
        return self.self_conditioning(active_hidden, signal)

    def _native_decoder_loss(self, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return self.output_head.weight.sum() * 0.0 + 2.0

    def _native_encoder_ar_loss(self, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return self.output_head.weight.sum() * 0.0 + 3.0


def _native_inputs() -> dict[str, object]:
    ids = torch.tensor([[1, 2, 3, 4]])
    return {
        "noisy_input_ids": ids,
        "clean_input_ids": ids,
        "labels": ids,
        "scored_mask": torch.ones_like(ids, dtype=torch.bool),
        "encoder_valid_mask": None,
        "decoder_position_ids": torch.arange(4),
        "decoder_valid_mask": None,
        "decoder_block_ids": torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        "response_start": 0,
        "block_token_counts": torch.full((1, 2), 2, dtype=torch.int64),
        "valid_block_mask": torch.ones((1, 2), dtype=torch.bool),
        "self_conditioning_mask": torch.tensor([True]),
        "self_conditioning_execute_all": False,
        "encoder_loss_weight": 0.5,
        "self_conditioning_row_chunk_size": 2,
        "self_conditioning_vocab_chunk_size": 4,
    }


def test_native_response_masks_use_relative_blocks_and_absolute_clean_bounds() -> None:
    model = _NativeExecutionHarness()
    model.block_size = 4
    layer = SimpleNamespace(
        layer_type="full_attention",
        attention=SimpleNamespace(sliding_window=None),
    )
    blocks = torch.tensor([0, 0, 0, 0, 1, -1, -1, -1], dtype=torch.int32)

    local, clean = model._native_attention_masks(
        query_positions=torch.arange(8),
        clean_positions=torch.arange(8),
        active_key_positions=torch.arange(8),
        query_is_clean=False,
        layer_ops=layer,
        pure_context=False,
        query_block_ids=blocks,
        active_key_block_ids=blocks,
        decoder_response_start=3,
    )

    torch.testing.assert_close(local.query_blocks, blocks)
    torch.testing.assert_close(local.active_blocks, blocks)
    torch.testing.assert_close(
        clean.query_clean_bounds,
        torch.tensor(
            [[0, 3], [0, 3], [0, 3], [0, 3], [0, 7], [0, 0], [0, 0], [0, 0]],
            dtype=torch.int32,
        ),
    )


def test_native_execution_reuses_one_clean_stream_with_detached_first_pass() -> None:
    model = _NativeExecutionHarness()
    output = model.forward_diffusiongemma_native(**_native_inputs())

    assert isinstance(output, DiffusionGemmaNativeOutput)
    assert model.calls == [
        ("clean", True, True),
        ("self_conditioning", False, False),
        ("decode", False, False),
        ("self_conditioning", True, True),
        ("decode", True, True),
    ]
    torch.testing.assert_close(output.loss, torch.tensor(3.5))
    torch.testing.assert_close(output.decoder_loss, torch.tensor(2.0))
    torch.testing.assert_close(output.encoder_loss, torch.tensor(3.0))


def test_native_zero_prefix_execution_preserves_original_decoder_hot_path() -> None:
    model = _NativeExecutionHarness()

    model.forward_diffusiongemma_native(**_native_inputs())

    assert model.decode_layout_metadata == [(None, None), (None, None)]


def test_native_execution_does_not_copy_single_example_clean_kv(monkeypatch) -> None:
    model = _NativeExecutionHarness()

    def _unexpected_copy(*args, **kwargs):
        del args, kwargs
        raise AssertionError("single-example detached pass copied the clean K/V stream")

    monkeypatch.setattr(
        _DiffusionGemmaCleanStream,
        "index_select_batch",
        _unexpected_copy,
    )

    model.forward_diffusiongemma_native(**_native_inputs())


def test_native_execution_skips_detached_pass_when_no_example_uses_it() -> None:
    model = _NativeExecutionHarness()
    inputs = _native_inputs()
    inputs["self_conditioning_mask"] = torch.tensor([False])
    model.forward_diffusiongemma_native(**inputs)

    assert model.calls == [
        ("clean", True, True),
        ("self_conditioning", True, False),
        ("decode", True, True),
    ]


def test_native_execution_pads_discarded_detached_rows_for_ep_shape_safety() -> None:
    model = _NativeExecutionHarness()
    inputs = _native_inputs()
    inputs["self_conditioning_mask"] = torch.tensor([False])
    inputs["self_conditioning_execution_count"] = 1

    model.forward_diffusiongemma_native(**inputs)

    assert model.calls == [
        ("clean", True, True),
        ("self_conditioning", False, False),
        ("decode", False, False),
        ("self_conditioning", True, False),
        ("decode", True, True),
    ]


def test_native_execution_runs_every_detached_row_for_nonzero_lora_dropout() -> None:
    model = _NativeExecutionHarness()
    inputs = _native_inputs()
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    inputs.update(
        {
            "noisy_input_ids": ids,
            "clean_input_ids": ids,
            "labels": ids,
            "scored_mask": torch.ones_like(ids, dtype=torch.bool),
            "block_token_counts": torch.full((2, 2), 2, dtype=torch.int64),
            "valid_block_mask": torch.ones((2, 2), dtype=torch.bool),
            "self_conditioning_mask": torch.tensor([False, True]),
            "self_conditioning_execution_count": 2,
            "self_conditioning_execute_all": True,
        }
    )

    model.forward_diffusiongemma_native(**inputs)

    assert model.decode_batch_sizes == [2, 2]
    assert model.last_self_conditioning_signal is not None
    assert not bool(model.last_self_conditioning_signal[0].any())
    assert bool(model.last_self_conditioning_signal[1].any())


def test_native_stream_positions_can_shard_shorter_response_than_clean_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _NativeExecutionHarness()
    model.runtime.kv_backend = "ring"
    model.runtime.context_attention_size = 2
    model.runtime.context_parallel_rank = 0
    model.runtime.block_parallel_size = 1
    monkeypatch.setattr(
        model,
        "_local_clean_positions",
        lambda device: torch.tensor([0, 1], device=device),
    )
    monkeypatch.setattr(
        model,
        "_packed_layout",
        lambda device: (_ for _ in ()).throw(
            AssertionError("compact response reused the full-sequence layout")
        ),
    )

    active, clean, pure_context = (
        NemotronLabsDiffusionPackedBlockDiffusionModel._native_stream_positions(
            model,
            torch.device("cpu"),
            decoder_length=2,
        )
    )

    torch.testing.assert_close(active, torch.tensor([0]))
    torch.testing.assert_close(clean, torch.tensor([0, 1]))
    assert pure_context is True


def test_native_soft_embeddings_use_the_released_embedding_scale(monkeypatch) -> None:
    model = _NativeExecutionHarness()
    model.soft_embedding_scale = 3.0
    observed: dict[str, float] = {}

    def _streaming_soft_embedding(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del weight
        observed["embedding_scale"] = float(kwargs["embedding_scale"])
        return torch.zeros_like(hidden)

    monkeypatch.setattr(
        "dllm_parallel.core.models.backbones.nemotron.model.streaming_soft_embedding",
        _streaming_soft_embedding,
    )
    NemotronLabsDiffusionPackedBlockDiffusionModel._native_apply_self_conditioning(
        model,
        torch.ones((1, 4, 3)),
        torch.ones((1, 4, 3)),
        torch.tensor([True]),
        row_chunk_size=2,
        vocab_chunk_size=4,
    )

    assert observed["embedding_scale"] == 3.0


def test_released_soft_embedding_scale_does_not_reuse_input_scaling() -> None:
    assert _soft_embedding_scale(
        SimpleNamespace(scalar_embed_scale=4.0),
        None,
    ) == pytest.approx(4.0)
    assert _soft_embedding_scale(SimpleNamespace(), 2.0) == pytest.approx(2.0)


def test_native_training_requires_full_activation_checkpointing() -> None:
    model = _NativeExecutionHarness()
    model.train()
    model.activation_checkpointing_scope = "mlp"

    with pytest.raises(RuntimeError, match="full activation checkpointing"):
        model.forward_diffusiongemma_native(**_native_inputs())


def test_native_forward_dispatches_through_module_call() -> None:
    model = _NativeExecutionHarness()
    output = model(objective_mode="diffusiongemma_native_sft", **_native_inputs())
    assert isinstance(output, DiffusionGemmaNativeOutput)


def test_native_pure_cp_masks_keep_full_canvas_metadata_for_compact_queries() -> None:
    model = _NativeExecutionHarness()
    layer = SimpleNamespace(
        layer_type="full_attention",
        attention=SimpleNamespace(sliding_window=None),
    )

    local, clean = model._native_attention_masks(
        query_positions=torch.tensor([0, 2]),
        clean_positions=torch.tensor([0, 1]),
        active_key_positions=torch.tensor([0, 2]),
        query_is_clean=False,
        layer_ops=layer,
        pure_context=True,
    )

    torch.testing.assert_close(
        local.query_blocks,
        torch.tensor([0, 0, 1, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(local.active_blocks, local.query_blocks)
    torch.testing.assert_close(
        clean.query_clean_bounds[:, 1],
        torch.tensor([0, 0, 2, 2], dtype=torch.int32),
    )


def test_native_fused_masks_reuse_static_tile_metadata_across_decoder_passes() -> None:
    model = _NativeExecutionHarness()
    model.runtime.block_parallel_size = 2
    model._layer_attention_mask_cache = {}
    layer = SimpleNamespace(
        layer_type="full_attention",
        attention=SimpleNamespace(sliding_window=None),
    )
    kwargs = {
        "query_positions": torch.tensor([0, 2]),
        "clean_positions": torch.tensor([0, 1]),
        "active_key_positions": torch.tensor([0, 2]),
        "query_is_clean": False,
        "layer_ops": layer,
        "pure_context": False,
    }

    first = model._native_attention_masks(**kwargs)
    first[1].flex_cache[("compiled-plan",)] = object()
    second = model._native_attention_masks(**kwargs)

    assert second[0] is first[0]
    assert second[1] is first[1]
    assert ("compiled-plan",) in second[1].flex_cache


def test_native_pure_cp_masks_do_not_reuse_fused_tile_metadata() -> None:
    model = _NativeExecutionHarness()
    model.runtime.block_parallel_size = 1
    model._layer_attention_mask_cache = {}
    layer = SimpleNamespace(
        layer_type="full_attention",
        attention=SimpleNamespace(sliding_window=None),
    )
    kwargs = {
        "query_positions": torch.tensor([0, 2]),
        "clean_positions": torch.tensor([0, 1]),
        "active_key_positions": torch.tensor([0, 2]),
        "query_is_clean": False,
        "layer_ops": layer,
        "pure_context": True,
    }

    first = model._native_attention_masks(**kwargs)
    second = model._native_attention_masks(**kwargs)

    assert second[0] is not first[0]
    assert second[1] is not first[1]
    assert model._layer_attention_mask_cache == {}


@pytest.mark.parametrize(
    (
        "kv_backend",
        "context_attention_size",
        "block_parallel_size",
        "context_parallel_rank",
        "expected_pure_context",
        "expected_active",
    ),
    (
        ("ring", 2, 1, 0, True, [0, 2]),
        ("ring", 2, 1, 1, True, [1, 3]),
        ("ring", 2, 2, 0, False, [0, 2]),
        ("ring", 1, 1, 0, False, [0, 2]),
        ("replicated", 1, 1, 0, False, [0, 2]),
    ),
)
def test_native_stream_positions_identify_pure_cp_without_bp_metadata(
    monkeypatch: pytest.MonkeyPatch,
    kv_backend: str,
    context_attention_size: int,
    block_parallel_size: int,
    context_parallel_rank: int,
    expected_pure_context: bool,
    expected_active: list[int],
) -> None:
    model = _NativeExecutionHarness()
    model.runtime.kv_backend = kv_backend
    model.runtime.context_attention_size = context_attention_size
    model.runtime.block_parallel_size = block_parallel_size
    model.runtime.context_parallel_rank = context_parallel_rank
    layout = SimpleNamespace(
        active_positions=torch.tensor([0, 2]),
        clean_positions=torch.tensor([0, 1]),
    )
    monkeypatch.setattr(model, "_packed_layout", lambda device: layout)

    active, clean, pure_context = (
        NemotronLabsDiffusionPackedBlockDiffusionModel._native_stream_positions(
            model,
            torch.device("cpu"),
        )
    )

    torch.testing.assert_close(active, torch.tensor(expected_active))
    if kv_backend == "replicated":
        torch.testing.assert_close(clean, torch.arange(model.seq_len))
    else:
        torch.testing.assert_close(clean, layout.clean_positions)
    assert pure_context is expected_pure_context


class _CheckpointHarness(NemotronLabsDiffusionPackedBlockDiffusionModel):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.layers = (object(),)
        self.norm = torch.nn.Identity()
        self.activation_checkpointing = True
        self.activation_checkpointing_scope = "full"
        self._adapter_checkpointing = True
        self._requires_moe_checkpoint_context = False
        self.runtime = SimpleNamespace(cp_bp_policy=None)
        self.weight = torch.nn.Parameter(torch.ones(1, 2, 3))

    def _native_embed_positions(self, input_ids, positions):
        del input_ids
        return self.weight.expand(1, int(positions.numel()), 3)

    def _position_embeddings_for_layer(self, *args, **kwargs):
        del args, kwargs
        return None, None

    def _native_attention_masks(self, **kwargs):
        del kwargs
        return object(), object()

    def _native_clean_layer_forward(
        self,
        layer_ops,
        hidden_states,
        **kwargs,
    ):
        del layer_ops, kwargs
        key = hidden_states.unsqueeze(2)
        return hidden_states + 1.0, key, key * 2.0

    def _native_active_layer_forward(
        self,
        layer_ops,
        hidden_states,
        clean_key,
        clean_value,
        **kwargs,
    ):
        del layer_ops, clean_value, kwargs
        return hidden_states + clean_key[:, : hidden_states.shape[1], 0]


class _SequenceParallelCheckpointHarness(_CheckpointHarness):
    def __init__(self) -> None:
        super().__init__()
        self.sequence_parallel = True
        self.runtime.tensor_parallel_size = 2
        self.sp_calls: list[tuple[str, bool | None]] = []

    def _native_scatter_stream(self, hidden_states, meta):
        self.sp_calls.append(("scatter", None))
        return hidden_states.reshape(meta.total_rows, meta.hidden_size)

    def _native_gather_stream(
        self,
        hidden_shard,
        meta,
        *,
        reduce_scatter_grad,
    ):
        self.sp_calls.append(("gather", bool(reduce_scatter_grad)))
        return hidden_shard.reshape(
            meta.batch_size,
            meta.packed_len,
            meta.hidden_size,
        )

    def _native_clean_layer_forward_sequence_parallel(
        self,
        layer_ops,
        hidden_shard,
        *,
        meta,
        **kwargs,
    ):
        del layer_ops, meta, kwargs
        self.sp_calls.append(("clean_layer", None))
        key = hidden_shard.reshape(1, 2, 1, 3)
        return hidden_shard + 1.0, key, key * 2.0

    def _native_active_layer_forward_sequence_parallel(
        self,
        layer_ops,
        hidden_shard,
        clean_key,
        clean_value,
        *,
        meta,
        **kwargs,
    ):
        del layer_ops, clean_value, meta, kwargs
        self.sp_calls.append(("active_layer", None))
        return hidden_shard + clean_key.reshape_as(hidden_shard)


class _NativeSequenceParallelLayerHarness(
    NemotronLabsDiffusionPackedBlockDiffusionModel
):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self._te_packed_by_layer_id = {}
        self.projected: list[tuple[torch.Tensor, object]] = []
        self.attention_clean_key: torch.Tensor | None = None
        self.finished_meta: object | None = None

    def _project_qkv_bshd_sequence_parallel(
        self,
        layer_ops,
        *,
        hidden_shard,
        meta,
        position_embeddings,
        cache_position,
    ):
        del layer_ops, position_embeddings, cache_position
        self.projected.append((hidden_shard, meta))
        full = hidden_shard.repeat(2, 1)[: meta.total_rows].reshape(1, 2, 1, 3)
        return full, full + 1.0, full + 2.0, None

    def _native_attention_from_projected(self, *args, **kwargs):
        del args
        self.attention_clean_key = kwargs["clean_key"]
        assert kwargs["sequence_parallel_meta"] is not None
        return kwargs["hidden_states"]

    def _native_finish_layer(
        self,
        layer_ops,
        *,
        residual,
        attention_output,
        clean_only,
        sequence_parallel_meta=None,
    ):
        del layer_ops, clean_only
        self.finished_meta = sequence_parallel_meta
        return residual + attention_output


def test_native_clean_and_trainable_decoder_layers_keep_full_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dllm_parallel.core.models.backbones.nemotron.model as module

    model = _CheckpointHarness()
    calls = 0

    def checkpoint(function, *args, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        return function(*args)

    monkeypatch.setattr(module, "checkpoint", checkpoint)
    positions = torch.tensor([0, 1])
    clean = model._native_encode_clean(torch.tensor([[1, 2]]), positions)
    active = torch.ones((1, 2, 3), requires_grad=True)
    with torch.no_grad():
        model._native_decode_active(
            active, positions, clean.detached(), pure_context=False
        )
    model._native_decode_active(active, positions, clean, pure_context=False)

    assert calls == 2


def test_native_sequence_parallel_shards_each_stream_between_layers() -> None:
    model = _SequenceParallelCheckpointHarness()
    positions = torch.tensor([0, 1])

    clean = model._native_encode_clean(torch.tensor([[1, 2]]), positions)
    active = model._native_decode_active(
        torch.ones((1, 2, 3), requires_grad=True),
        positions,
        clean,
        pure_context=False,
    )

    assert clean.hidden.shape == (1, 2, 3)
    assert active.shape == (1, 2, 3)
    assert model.sp_calls == [
        ("scatter", None),
        ("clean_layer", None),
        ("gather", False),
        ("scatter", None),
        ("active_layer", None),
        ("gather", False),
    ]
    (clean.hidden.sum() + active.sum()).backward()
    assert model.weight.grad is not None
    assert bool(torch.isfinite(model.weight.grad).all())


def test_native_sp_layers_retain_full_row_local_head_clean_kv() -> None:
    model = _NativeSequenceParallelLayerHarness()
    layer = SimpleNamespace(
        layer=object(),
        input_layernorm=torch.nn.Identity(),
    )
    hidden_shard = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    meta = SimpleNamespace(total_rows=2)

    clean_hidden, clean_key, clean_value = (
        model._native_clean_layer_forward_sequence_parallel(
            layer,
            hidden_shard,
            meta=meta,
            position_embeddings=(None, None),
            clean_positions=torch.tensor([0, 1]),
            local_attn_mask=object(),
            global_attn_mask=object(),
            pure_context=False,
        )
    )
    active_hidden = model._native_active_layer_forward_sequence_parallel(
        layer,
        hidden_shard,
        clean_key,
        clean_value,
        meta=meta,
        position_embeddings=(None, None),
        active_positions=torch.tensor([0, 1]),
        local_attn_mask=object(),
        global_attn_mask=object(),
        pure_context=False,
    )

    assert clean_key.shape == (1, 2, 1, 3)
    assert clean_value.shape == (1, 2, 1, 3)
    assert model.attention_clean_key is clean_key
    assert len(model.projected) == 2
    assert all(states is hidden_shard for states, _ in model.projected)
    assert all(projected_meta is meta for _, projected_meta in model.projected)
    assert model.finished_meta is meta
    torch.testing.assert_close(clean_hidden, hidden_shard * 2.0)
    torch.testing.assert_close(active_hidden, hidden_shard * 2.0)


def test_native_sp_attention_pads_full_rows_before_output_reduce_scatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dllm_parallel.core.models.backbones.nemotron.model as module

    model = _NativeSequenceParallelLayerHarness()
    model.runtime = SimpleNamespace(kv_backend="replicated")
    projected: list[torch.Tensor] = []

    def attention(*args, **kwargs):
        del args, kwargs
        return torch.ones(1, 3, 1, 2)

    def output_projection(layer_ops, value):
        del layer_ops
        projected.append(value)
        return value[:2]

    monkeypatch.setattr(
        module,
        "replicated_block_denoising_attention_bshd",
        attention,
    )
    monkeypatch.setattr(model, "_attention_output_projection", output_projection)
    meta = SimpleNamespace(total_rows=3, padded_rows=4)
    query = torch.zeros(1, 3, 1, 2)
    output = (
        NemotronLabsDiffusionPackedBlockDiffusionModel._native_attention_from_projected(
            model,
            SimpleNamespace(attention=SimpleNamespace(scale=1.0)),
            hidden_states=torch.zeros(2, 6),
            query=query,
            active_key=query,
            active_value=query,
            clean_key=query[:, :0],
            clean_value=query[:, :0],
            output_gate=None,
            local_attn_mask=object(),
            global_attn_mask=object(),
            pure_context=False,
            sequence_parallel_meta=meta,
        )
    )

    assert projected[0].shape == (4, 2)
    torch.testing.assert_close(projected[0][-1], torch.zeros(2))
    assert output.shape == (2, 2)


def test_native_tp_only_does_not_enter_sequence_parallel_helpers() -> None:
    model = _CheckpointHarness()
    model.sequence_parallel = False
    model.runtime.tensor_parallel_size = 2

    def unexpected(*args, **kwargs):
        del args, kwargs
        raise AssertionError("TP-only native execution entered an SP helper")

    model._native_scatter_stream = unexpected
    model._native_gather_stream = unexpected
    positions = torch.tensor([0, 1])
    clean = model._native_encode_clean(torch.tensor([[1, 2]]), positions)
    active = model._native_decode_active(
        torch.ones((1, 2, 3), requires_grad=True),
        positions,
        clean,
        pure_context=False,
    )

    assert active.shape == (1, 2, 3)


def test_native_forward_accepts_tensor_sequence_parallel_runtime() -> None:
    model = _NativeExecutionHarness()
    model.sequence_parallel = True
    model.runtime.tensor_parallel_size = 2

    output = model.forward_diffusiongemma_native(**_native_inputs())

    assert isinstance(output, DiffusionGemmaNativeOutput)


def test_native_clean_stream_preserves_pure_cp_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _CheckpointHarness()
    observed: list[bool] = []
    original_masks = model._native_attention_masks

    def recording_masks(**kwargs):
        observed.append(bool(kwargs["pure_context"]))
        return original_masks(**kwargs)

    monkeypatch.setattr(model, "_native_attention_masks", recording_masks)
    positions = torch.tensor([0, 1])
    model._native_encode_clean(
        torch.tensor([[1, 2]]),
        positions,
        pure_context=True,
    )

    assert observed == [True]


def test_native_stream_sequence_parallel_metadata_is_independent() -> None:
    model = _NativeExecutionHarness()
    model.runtime.tensor_parallel_size = 4
    clean = torch.zeros(2, 5, 3)
    active = torch.zeros(2, 3, 3)

    clean_meta = model._native_sequence_parallel_meta(clean)
    active_meta = model._native_sequence_parallel_meta(active)

    assert (
        clean_meta.batch_size,
        clean_meta.packed_len,
        clean_meta.total_rows,
        clean_meta.padded_rows,
        clean_meta.active_len,
    ) == (2, 5, 10, 12, 5)
    assert (
        active_meta.batch_size,
        active_meta.packed_len,
        active_meta.total_rows,
        active_meta.padded_rows,
        active_meta.active_len,
    ) == (2, 3, 6, 8, 3)


def test_native_stream_scatter_pads_rows_without_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dllm_parallel.core.models.backbones.nemotron.model as module

    model = _NativeExecutionHarness()
    model.runtime.tensor_parallel_size = 2
    hidden = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
    meta = model._native_sequence_parallel_meta(hidden)
    observed: list[torch.Tensor] = []

    def recording_scatter(value, runtime):
        assert runtime is model.runtime
        observed.append(value.clone())
        return value[:3]

    monkeypatch.setattr(
        module, "scatter_to_sequence_parallel_region", recording_scatter
    )

    shard = model._native_scatter_stream(hidden, meta)

    torch.testing.assert_close(observed[0][:5], hidden.reshape(5, 3))
    torch.testing.assert_close(observed[0][5], torch.zeros(3))
    torch.testing.assert_close(shard, observed[0][:3])


@pytest.mark.parametrize("reduce_scatter_grad", [False, True])
def test_native_stream_gather_restores_batch_order_and_gradient_contract(
    monkeypatch: pytest.MonkeyPatch,
    reduce_scatter_grad: bool,
) -> None:
    import dllm_parallel.core.models.backbones.nemotron.model as module

    model = _NativeExecutionHarness()
    model.runtime.tensor_parallel_size = 2
    reference = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    meta = model._native_sequence_parallel_meta(reference)
    observed: list[tuple[int | None, bool]] = []

    def recording_gather(value, runtime, *, total_rows, reduce_scatter_grad):
        del value
        assert runtime is model.runtime
        observed.append((total_rows, reduce_scatter_grad))
        return reference.reshape(10, 3)

    monkeypatch.setattr(
        module, "gather_from_sequence_parallel_region", recording_gather
    )

    gathered = model._native_gather_stream(
        torch.empty(5, 3),
        meta,
        reduce_scatter_grad=reduce_scatter_grad,
    )

    torch.testing.assert_close(gathered, reference)
    assert observed == [(10, reduce_scatter_grad)]
