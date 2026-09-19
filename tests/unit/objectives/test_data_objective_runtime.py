from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dllm_parallel.core.data import (
    INDEXED_SUPERVISED_FORMAT,
    INDEXED_SUPERVISED_VERSION,
    BlendedPackedTokenDataRuntime,
    DataBatch,
    IndexedSupervisedTokenDataRuntime,
    PackedTokenDataRuntime,
    RandomTokenDataRuntime,
    tokenizer_vocabulary_sha256,
)
from dllm_parallel.core.objectives.runtime import (
    StandardBlockDiffusionObjectiveRuntime,
    TokenAccountingPolicy,
)
from dllm_parallel.core.objectives.training import build_standard_training_task
from dllm_parallel.data.indexed import PACKED_TOKEN_FORMAT, PACKED_TOKEN_VERSION


def _accounting() -> TokenAccountingPolicy:
    return TokenAccountingPolicy(
        global_batch_size=2,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        data_parallel_size=1,
        context_parallel_size=1,
        block_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel=False,
    )


def test_random_data_runtime_restores_generator_state() -> None:
    runtime = RandomTokenDataRuntime(
        batch_size=2,
        seq_len=4,
        vocab_size=17,
        sample_vocab_size=None,
        mask_token_id=16,
        device=torch.device("cpu"),
        seed=123,
    )
    first = runtime.next_batch().input_ids
    state = runtime.state_dict()
    expected = runtime.next_batch().input_ids

    restored = RandomTokenDataRuntime(
        batch_size=2,
        seq_len=4,
        vocab_size=17,
        sample_vocab_size=None,
        mask_token_id=16,
        device=torch.device("cpu"),
        seed=999,
    )
    restored.load_state_dict(state)

    assert first.shape == (2, 4)
    torch.testing.assert_close(restored.next_batch().input_ids, expected)


def test_packed_token_runtime_restores_cursor(tmp_path) -> None:
    path = tmp_path / "tokens.pt"
    torch.save(torch.arange(20, dtype=torch.long), path)
    runtime = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(path),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
    )
    first = runtime.next_batch().input_ids
    state = runtime.state_dict()
    second = runtime.next_batch().input_ids
    restored = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(path),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
    )
    restored.load_state_dict(state)

    torch.testing.assert_close(first, torch.tensor([[0, 1, 2], [3, 4, 5]]))
    torch.testing.assert_close(restored.next_batch().input_ids, second)


def test_packed_token_runtime_shards_batches_by_data_parallel_rank(tmp_path) -> None:
    path = tmp_path / "tokens.pt"
    torch.save(torch.arange(40, dtype=torch.long), path)
    rank0 = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(path),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
        data_parallel_rank=0,
        data_parallel_size=2,
    )
    rank1 = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(path),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
        data_parallel_rank=1,
        data_parallel_size=2,
    )

    torch.testing.assert_close(
        rank0.next_batch().input_ids,
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )
    torch.testing.assert_close(
        rank1.next_batch().input_ids,
        torch.tensor([[6, 7, 8], [9, 10, 11]]),
    )
    torch.testing.assert_close(
        rank0.next_batch().input_ids,
        torch.tensor([[12, 13, 14], [15, 16, 17]]),
    )
    torch.testing.assert_close(
        rank1.next_batch().input_ids,
        torch.tensor([[18, 19, 20], [21, 22, 23]]),
    )


def test_packed_token_runtime_loads_binary_int64_stream(tmp_path) -> None:
    path = tmp_path / "tokens.i64"
    expected = torch.arange(12, dtype=torch.int64)
    path.write_bytes(expected.numpy().tobytes())

    runtime = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(path),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        runtime.next_batch().input_ids,
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )


def test_packed_token_runtime_loads_binary_int32_stream(tmp_path) -> None:
    path = tmp_path / "tokens.i32"
    expected = torch.arange(12, dtype=torch.int32)
    path.write_bytes(expected.numpy().tobytes())

    runtime = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(path),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
    )

    batch = runtime.next_batch().input_ids
    assert runtime.tokens.dtype == torch.int32
    assert batch.dtype == torch.long
    torch.testing.assert_close(
        batch,
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )


def test_packed_token_runtime_loads_prepared_artifact_directory(tmp_path) -> None:
    root = tmp_path / "prepared"
    root.mkdir()
    expected = torch.arange(12, dtype=torch.int32)
    payload = expected.numpy().tobytes()
    (root / "tokens.i32").write_bytes(payload)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": PACKED_TOKEN_FORMAT,
                "version": PACKED_TOKEN_VERSION,
                "metadata": {"dataset_fingerprint": "a" * 64},
                "sample_count": 3,
                "token_count": 12,
                "files": {
                    "tokens.i32": {
                        "dtype": "int32",
                        "elements": 12,
                        "bytes": len(payload),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    runtime = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(root),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
    )
    first = runtime.next_batch().input_ids
    state = runtime.state_dict()
    expected_next = runtime.next_batch().input_ids
    restored = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(root),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
    )
    restored.load_state_dict(state)

    torch.testing.assert_close(first, torch.tensor([[0, 1, 2], [3, 4, 5]]))
    torch.testing.assert_close(restored.next_batch().input_ids, expected_next)
    assert runtime.to_log_dict()["kind"] == "prepared_packed_tokens"


def test_packed_token_runtime_rejects_resume_from_different_prepared_artifact(
    tmp_path,
) -> None:
    def prepared(name: str, fingerprint: str) -> Path:
        root = tmp_path / name
        root.mkdir()
        payload = torch.arange(12, dtype=torch.int32).numpy().tobytes()
        (root / "tokens.i32").write_bytes(payload)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "format": PACKED_TOKEN_FORMAT,
                    "version": PACKED_TOKEN_VERSION,
                    "metadata": {"dataset_fingerprint": fingerprint},
                    "sample_count": 3,
                    "token_count": 12,
                    "files": {
                        "tokens.i32": {
                            "dtype": "int32",
                            "elements": 12,
                            "bytes": len(payload),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return root

    first = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(prepared("first", "a" * 64)),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
    )
    second = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(prepared("second", "b" * 64)),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="dataset fingerprint"):
        second.load_state_dict(first.state_dict())


def test_packed_token_runtime_rejects_mismatched_dp_resume_state(tmp_path) -> None:
    path = tmp_path / "tokens.pt"
    torch.save(torch.arange(20, dtype=torch.long), path)
    rank0 = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(path),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
        data_parallel_rank=0,
        data_parallel_size=2,
    )
    state = rank0.state_dict()
    rank1 = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(path),
        tokenizer=None,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
        data_parallel_rank=1,
        data_parallel_size=2,
    )

    try:
        rank1.load_state_dict(state)
    except RuntimeError as exc:
        assert "data_parallel_rank" in str(exc)
    else:
        raise AssertionError("expected mismatched DP rank to be rejected")


def test_blended_packed_token_runtime_restores_dataset_and_rng_state(tmp_path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    manifest = tmp_path / "manifest.json"
    torch.save(torch.arange(20, dtype=torch.long), first)
    torch.save(torch.arange(100, 120, dtype=torch.long), second)
    manifest.write_text(
        json.dumps(
            {
                "datasets": [
                    {"path": first.name, "weight": 1.0},
                    {"path": second.name, "weight": 3.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    runtime = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(manifest),
        tokenizer=None,
        batch_size=1,
        seq_len=2,
        device=torch.device("cpu"),
        blend_seed=17,
    )
    assert isinstance(runtime, BlendedPackedTokenDataRuntime)
    runtime.next_batch()
    state = runtime.state_dict()
    expected = runtime.next_batch().input_ids

    restored = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(manifest),
        tokenizer=None,
        batch_size=1,
        seq_len=2,
        device=torch.device("cpu"),
        blend_seed=999,
    )
    restored.load_state_dict(state)

    torch.testing.assert_close(restored.next_batch().input_ids, expected)


def test_objective_runtime_restores_corruption_rng_and_counts() -> None:
    objective = StandardBlockDiffusionObjectiveRuntime(
        mask_token_id=99,
        block_size=2,
        seq_len=4,
        device=torch.device("cpu"),
        seed=7,
        token_accounting=_accounting(),
    )
    x0 = torch.arange(8, dtype=torch.long).reshape(2, 4)
    first = objective.corrupt(x0)
    state = objective.state_dict()
    expected = objective.corrupt(x0)

    restored = StandardBlockDiffusionObjectiveRuntime(
        mask_token_id=99,
        block_size=2,
        seq_len=4,
        device=torch.device("cpu"),
        seed=999,
        token_accounting=_accounting(),
    )
    restored.load_state_dict(state)
    observed = restored.corrupt(x0)

    assert first.active_tokens > 0
    assert objective.loss_tokens_seen >= first.active_tokens
    torch.testing.assert_close(observed.noisy_input_ids, expected.noisy_input_ids)
    torch.testing.assert_close(observed.labels, expected.labels)
    torch.testing.assert_close(observed.diffusion_times, expected.diffusion_times)
    torch.testing.assert_close(observed.noise_levels, expected.noise_levels)
    torch.testing.assert_close(observed.loss_weights, expected.loss_weights)


def test_objective_runtime_accepts_bounded_block_aligned_lengths() -> None:
    objective = StandardBlockDiffusionObjectiveRuntime(
        mask_token_id=99,
        block_size=2,
        seq_len=8,
        device=torch.device("cpu"),
        seed=7,
        token_accounting=_accounting(),
    )

    short = objective.corrupt(torch.arange(8).reshape(2, 4))
    maximum = objective.corrupt(torch.arange(16).reshape(2, 8))

    assert short.diffusion_times.shape == (2, 2)
    assert maximum.diffusion_times.shape == (2, 4)
    with pytest.raises(ValueError, match="block_size"):
        objective.corrupt(torch.arange(6).reshape(2, 3))
    with pytest.raises(ValueError, match="configured maximum"):
        objective.corrupt(torch.arange(20).reshape(2, 10))


def test_bounded_objective_preserves_fixed_shape_rng_and_outputs() -> None:
    exact = StandardBlockDiffusionObjectiveRuntime(
        mask_token_id=99,
        block_size=2,
        seq_len=4,
        device=torch.device("cpu"),
        seed=7,
        token_accounting=_accounting(),
    )
    bounded = StandardBlockDiffusionObjectiveRuntime(
        mask_token_id=99,
        block_size=2,
        seq_len=8,
        device=torch.device("cpu"),
        seed=7,
        token_accounting=_accounting(),
    )
    input_ids = torch.arange(8).reshape(2, 4)

    expected = exact.corrupt(input_ids)
    observed = bounded.corrupt(input_ids)

    torch.testing.assert_close(observed.noisy_input_ids, expected.noisy_input_ids)
    torch.testing.assert_close(observed.labels, expected.labels)
    torch.testing.assert_close(observed.diffusion_times, expected.diffusion_times)
    torch.testing.assert_close(observed.noise_levels, expected.noise_levels)
    torch.testing.assert_close(observed.loss_weights, expected.loss_weights)
    assert observed.active_tokens.item() == expected.active_tokens.item()


class _SupervisedTokenizer:
    mask_token_id = 127
    chat_template = "template"

    def __len__(self) -> int:
        return 128

    def get_vocab(self) -> dict[str, int]:
        return {str(index): index for index in range(128)}


def _write_indexed_supervised_dataset(tmp_path) -> None:
    root = tmp_path / "indexed"
    root.mkdir()
    samples = (
        [
            (torch.arange(4, dtype=torch.int32) + offset, 2, 100 + sample)
            for sample, offset in enumerate((0, 10, 20, 30))
        ]
        + [
            (torch.arange(8, dtype=torch.int32) + offset, 4, 200 + sample)
            for sample, offset in enumerate((40, 50, 60, 70))
        ]
        + [(torch.arange(12, dtype=torch.int32) + 80, 8, 300)]
    )
    tokens = torch.cat([sample[0] for sample in samples])
    records = []
    offset = 0
    for sample_tokens, supervision_start, sample_id in samples:
        records.append(
            (offset, int(sample_tokens.numel()), supervision_start, sample_id)
        )
        offset += int(sample_tokens.numel())
    index = torch.tensor(records, dtype=torch.int64)
    token_bytes = tokens.numpy().tobytes()
    index_bytes = index.numpy().tobytes()
    (root / "tokens.i32").write_bytes(token_bytes)
    (root / "index.i64").write_bytes(index_bytes)
    tokenizer = _SupervisedTokenizer()
    metadata = {
        "tokenizer_size": len(tokenizer),
        "tokenizer_vocabulary_sha256": tokenizer_vocabulary_sha256(tokenizer),
        "mask_token_id": tokenizer.mask_token_id,
        "chat_template_sha256": hashlib.sha256(b"template").hexdigest(),
        "semantics": "assistant_only_sft",
        "block_size": 2,
        "contains_mask_token": False,
        "dataset_fingerprint": "indexed-test-dataset",
    }
    manifest = {
        "format": INDEXED_SUPERVISED_FORMAT,
        "version": INDEXED_SUPERVISED_VERSION,
        "metadata": metadata,
        "sample_count": len(samples),
        "token_count": int(tokens.numel()),
        "files": {
            "tokens.i32": {
                "dtype": "int32",
                "elements": int(tokens.numel()),
                "bytes": len(token_bytes),
            },
            "index.i64": {
                "dtype": "int64",
                "columns": [
                    "token_offset",
                    "sequence_length",
                    "supervision_start",
                    "sample_id",
                ],
                "elements": int(index.numel()),
                "bytes": len(index_bytes),
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_indexed_supervised_runtime_mmaps_variable_length_dp_batches(
    tmp_path,
) -> None:
    _write_indexed_supervised_dataset(tmp_path)
    runtimes = [
        PackedTokenDataRuntime.from_dataset_path(
            dataset_path=str(tmp_path / "indexed"),
            tokenizer=_SupervisedTokenizer(),
            mask_token_id=127,
            batch_size=1,
            seq_len=8,
            block_size=2,
            context_parallel_size=1,
            block_parallel_size=1,
            device=torch.device("cpu"),
            data_parallel_rank=rank,
            data_parallel_size=2,
            blend_seed=17,
            shuffle=False,
        )
        for rank in range(2)
    ]
    assert all(
        isinstance(runtime, IndexedSupervisedTokenDataRuntime) for runtime in runtimes
    )
    first = [runtime.next_batch() for runtime in runtimes]
    assert [tuple(batch.input_ids.shape) for batch in first] == [(1, 4), (1, 4)]
    assert first[0].sample_ids.item() != first[1].sample_ids.item()
    assert all(batch.loss_mask[:, 2:].all() for batch in first)
    state = runtimes[0].state_dict()
    expected = runtimes[0].next_batch()
    restored = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=str(tmp_path / "indexed"),
        tokenizer=_SupervisedTokenizer(),
        mask_token_id=127,
        batch_size=1,
        seq_len=8,
        block_size=2,
        context_parallel_size=1,
        block_parallel_size=1,
        device=torch.device("cpu"),
        data_parallel_rank=0,
        data_parallel_size=2,
        blend_seed=17,
        shuffle=False,
    )
    restored.load_state_dict(state)
    observed = restored.next_batch()
    torch.testing.assert_close(observed.input_ids, expected.input_ids)
    torch.testing.assert_close(observed.loss_mask, expected.loss_mask)
    assert restored.to_log_dict()["eligible_sample_count"] == 8


def test_indexed_supervised_runtime_enforces_topology_and_restart_contract(
    tmp_path,
) -> None:
    _write_indexed_supervised_dataset(tmp_path)
    root = str(tmp_path / "indexed")
    fused = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=root,
        tokenizer=_SupervisedTokenizer(),
        mask_token_id=127,
        batch_size=1,
        seq_len=8,
        block_size=2,
        context_parallel_size=2,
        block_parallel_size=2,
        device=torch.device("cpu"),
        shuffle=False,
    )
    # BP workers own complete blocks independently, so the four-token and
    # eight-token samples are both eligible for BP=2.
    assert fused.to_log_dict()["eligible_sample_count"] == 8

    pure_cp = PackedTokenDataRuntime.from_dataset_path(
        dataset_path=root,
        tokenizer=_SupervisedTokenizer(),
        mask_token_id=127,
        batch_size=1,
        seq_len=12,
        block_size=2,
        context_parallel_size=3,
        block_parallel_size=1,
        device=torch.device("cpu"),
        shuffle=False,
    )
    assert pure_cp.to_log_dict()["eligible_sample_count"] == 1

    incompatible_state = fused.state_dict()
    incompatible_state["block_parallel_size"] = 1
    with pytest.raises(RuntimeError, match="block_parallel_size"):
        fused.load_state_dict(incompatible_state)


def test_indexed_supervised_runtime_can_match_cp_and_fused_sample_sets(
    tmp_path,
) -> None:
    _write_indexed_supervised_dataset(tmp_path)
    root = str(tmp_path / "indexed")
    common = {
        "dataset_path": root,
        "tokenizer": _SupervisedTokenizer(),
        "mask_token_id": 127,
        "batch_size": 1,
        "seq_len": 8,
        "minimum_sequence_length": 8,
        "block_size": 2,
        "context_parallel_size": 2,
        "device": torch.device("cpu"),
        "blend_seed": 19,
        "shuffle": True,
    }
    pure_cp = PackedTokenDataRuntime.from_dataset_path(
        **common,
        block_parallel_size=1,
    )
    fused = PackedTokenDataRuntime.from_dataset_path(
        **common,
        block_parallel_size=2,
    )

    torch.testing.assert_close(pure_cp.eligible_indices, fused.eligible_indices)
    assert pure_cp.to_log_dict()["eligible_sample_count"] == 4
    assert fused.to_log_dict()["eligible_sample_count"] == 4
    torch.testing.assert_close(
        pure_cp.next_batch().sample_ids,
        fused.next_batch().sample_ids,
    )


def test_indexed_supervised_runtime_round_robins_trajectory_groups() -> None:
    sample_lengths = [4, 4, 4, 4, 8, 8, 8, 8]
    sample_groups = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.int64)
    sample_ids = [100 + index for index in range(len(sample_lengths))]
    tokens = torch.arange(sum(sample_lengths), dtype=torch.int32)
    records = []
    offset = 0
    for length, sample_id in zip(sample_lengths, sample_ids, strict=True):
        records.append((offset, length, length // 2, sample_id))
        offset += length
    runtime = IndexedSupervisedTokenDataRuntime(
        tokens=tokens,
        index=torch.tensor(records, dtype=torch.int64),
        group_ids=sample_groups,
        batch_size=1,
        max_seq_len=8,
        block_size=2,
        context_parallel_size=1,
        block_parallel_size=1,
        device=torch.device("cpu"),
        source="group-round-test",
        data_parallel_rank=0,
        data_parallel_size=1,
        seed=2026,
        shuffle=True,
        minimum_sequence_length=None,
        dataset_fingerprint="group-round-test",
    )
    id_to_group = {
        sample_id: int(group)
        for sample_id, group in zip(sample_ids, sample_groups, strict=True)
    }

    first_round = [runtime.next_batch() for _ in range(4)]
    observed_groups = [id_to_group[int(batch.sample_ids.item())] for batch in first_round]
    observed_lengths = [int(batch.input_ids.shape[1]) for batch in first_round]

    assert len(set(observed_groups)) == 4
    assert set(observed_lengths) == {4, 8}


def test_indexed_supervised_runtime_globally_interleaves_length_batches() -> None:
    sample_lengths = [4] * 12 + [8] * 12 + [12] * 12
    tokens = torch.arange(sum(sample_lengths), dtype=torch.int32)
    records = []
    offset = 0
    for sample_id, length in enumerate(sample_lengths):
        records.append((offset, length, length // 2, 100 + sample_id))
        offset += length
    runtime = IndexedSupervisedTokenDataRuntime(
        tokens=tokens,
        index=torch.tensor(records, dtype=torch.int64),
        batch_size=1,
        max_seq_len=12,
        block_size=2,
        context_parallel_size=1,
        block_parallel_size=1,
        device=torch.device("cpu"),
        source="length-interleave-test",
        data_parallel_rank=0,
        data_parallel_size=1,
        seed=2026,
        shuffle=True,
        minimum_sequence_length=None,
        dataset_fingerprint="length-interleave-test",
    )

    prefix_lengths = [
        int(runtime.next_batch().input_ids.shape[1]) for _ in range(12)
    ]

    assert len(set(prefix_lengths)) == 3


def _trajectory_step_runtime(*, seed: int = 2026) -> IndexedSupervisedTokenDataRuntime:
    sample_lengths = [4, 8, 4, 6, 8]
    sample_groups = torch.tensor([11, 11, 22, 22, 22], dtype=torch.int64)
    tokens = torch.arange(sum(sample_lengths), dtype=torch.int32)
    records = []
    offset = 0
    for sample_id, length in enumerate(sample_lengths, start=100):
        records.append((offset, length, length // 2, sample_id))
        offset += length
    return IndexedSupervisedTokenDataRuntime(
        tokens=tokens,
        index=torch.tensor(records, dtype=torch.int64),
        group_ids=sample_groups,
        batch_size=1,
        max_seq_len=8,
        block_size=2,
        context_parallel_size=1,
        block_parallel_size=1,
        device=torch.device("cpu"),
        source="trajectory-step-test",
        data_parallel_rank=0,
        data_parallel_size=1,
        seed=seed,
        shuffle=True,
        minimum_sequence_length=None,
        dataset_fingerprint="trajectory-step-test",
        optimizer_step_unit="trajectory",
    )


def test_indexed_supervised_runtime_returns_one_complete_trajectory_per_step() -> None:
    runtime = _trajectory_step_runtime()
    id_to_group = {100: 11, 101: 11, 102: 22, 103: 22, 104: 22}

    first = runtime.next_optimizer_step_batches(1)
    second = runtime.next_optimizer_step_batches(1)
    observed = [first, second]

    assert sorted(len(step) for step in observed) == [2, 3]
    assert {
        id_to_group[int(batch.sample_ids.item())]
        for step in observed
        for batch in step
    } == {11, 22}
    for step in observed:
        assert len(
            {id_to_group[int(batch.sample_ids.item())] for batch in step}
        ) == 1
    assert runtime.to_log_dict()["epoch_optimizer_steps"] == 2
    assert runtime.to_log_dict()["samples_consumed"] == 5


def test_indexed_trajectory_step_runtime_restores_exact_group_boundary() -> None:
    runtime = _trajectory_step_runtime(seed=17)
    runtime.next_optimizer_step_batches(1)
    state = runtime.state_dict()
    expected = [
        int(batch.sample_ids.item())
        for batch in runtime.next_optimizer_step_batches(1)
    ]

    restored = _trajectory_step_runtime(seed=17)
    restored.load_state_dict(state)
    observed = [
        int(batch.sample_ids.item())
        for batch in restored.next_optimizer_step_batches(1)
    ]

    assert observed == expected


def test_indexed_trajectory_steps_reject_static_accumulation() -> None:
    runtime = _trajectory_step_runtime()

    with pytest.raises(RuntimeError, match="gradient_accumulation_steps=1"):
        runtime.next_optimizer_step_batches(2)


def test_indexed_trajectory_steps_reject_partially_eligible_groups() -> None:
    tokens = torch.arange(16, dtype=torch.int32)
    index = torch.tensor(
        [(0, 4, 2, 100), (4, 12, 6, 101)],
        dtype=torch.int64,
    )

    with pytest.raises(RuntimeError, match="partially eligible group"):
        IndexedSupervisedTokenDataRuntime(
            tokens=tokens,
            index=index,
            group_ids=torch.tensor([1, 1], dtype=torch.int64),
            batch_size=1,
            max_seq_len=8,
            block_size=2,
            context_parallel_size=1,
            block_parallel_size=1,
            device=torch.device("cpu"),
            source="partial-group-test",
            data_parallel_rank=0,
            data_parallel_size=1,
            seed=1,
            shuffle=False,
            minimum_sequence_length=None,
            dataset_fingerprint="partial-group-test",
            optimizer_step_unit="trajectory",
        )


def test_indexed_trajectory_steps_reject_data_parallel_execution() -> None:
    tokens = torch.arange(8, dtype=torch.int32)
    index = torch.tensor(
        [(0, 4, 2, 100), (4, 4, 2, 101)],
        dtype=torch.int64,
    )

    with pytest.raises(RuntimeError, match="data_parallel_size=1"):
        IndexedSupervisedTokenDataRuntime(
            tokens=tokens,
            index=index,
            group_ids=torch.tensor([1, 2], dtype=torch.int64),
            batch_size=1,
            max_seq_len=4,
            block_size=2,
            context_parallel_size=1,
            block_parallel_size=1,
            device=torch.device("cpu"),
            source="trajectory-dp-test",
            data_parallel_rank=0,
            data_parallel_size=2,
            seed=1,
            shuffle=False,
            minimum_sequence_length=None,
            dataset_fingerprint="trajectory-dp-test",
            optimizer_step_unit="trajectory",
        )


def test_block_diffusion_corrupts_only_supervised_tokens() -> None:
    runtime = StandardBlockDiffusionObjectiveRuntime(
        mask_token_id=99,
        block_size=2,
        seq_len=4,
        device=torch.device("cpu"),
        seed=7,
        token_accounting=_accounting(),
        sampling_epsilon_min=1.0,
        sampling_epsilon_max=1.0,
        antithetic_sampling=False,
    )
    clean = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    supervision = torch.tensor([[False, False, True, True], [False, True, False, True]])
    corrupted = runtime.corrupt(clean, supervision_mask=supervision)
    torch.testing.assert_close(
        corrupted.noisy_input_ids,
        torch.where(supervision, torch.full_like(clean, 99), clean),
    )
    assert corrupted.valid_tokens == int(supervision.sum())
    assert int(corrupted.active_tokens) == int(supervision.sum())
    assert torch.equal(corrupted.labels.ne(-100), supervision)


def test_objective_runtime_uses_blockwise_loglinear_schedule() -> None:
    objective = StandardBlockDiffusionObjectiveRuntime(
        mask_token_id=99,
        block_size=2,
        seq_len=8,
        device=torch.device("cpu"),
        seed=11,
        token_accounting=_accounting(),
    )
    x0 = torch.arange(16, dtype=torch.long).reshape(2, 8)

    corrupted = objective.corrupt(x0)

    assert corrupted.diffusion_times.shape == (2, 4)
    assert corrupted.noise_levels.shape == (2, 4)
    assert corrupted.loss_weights.shape == (2, 4)
    assert torch.all(corrupted.diffusion_times >= 1.0e-3)
    assert torch.all(corrupted.diffusion_times <= 1.0)
    torch.testing.assert_close(
        corrupted.loss_weights,
        corrupted.diffusion_times.reciprocal(),
    )
    torch.testing.assert_close(
        corrupted.noise_levels,
        (-torch.log1p(-corrupted.diffusion_times)).clamp(
            max=-torch.log(torch.tensor(1.0e-3))
        ),
    )
    expected_labels = torch.where(
        corrupted.noisy_input_ids == 99,
        x0,
        torch.full_like(x0, -100),
    )
    torch.testing.assert_close(corrupted.labels, expected_labels)


def test_objective_runtime_matches_standard_antithetic_time_sampling() -> None:
    seed = 11
    batch_size = 2
    num_blocks = 4
    sample_count = batch_size * num_blocks
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = torch.rand(
        (1,),
        dtype=torch.float32,
        generator=generator,
    )
    offsets = torch.arange(sample_count, dtype=torch.float32).view(
        batch_size,
        num_blocks,
    )
    expected = (base + offsets / sample_count).remainder(1.0)
    expected = expected * (1.0 - 1.0e-3) + 1.0e-3
    objective = StandardBlockDiffusionObjectiveRuntime(
        mask_token_id=99,
        block_size=2,
        seq_len=8,
        device=torch.device("cpu"),
        seed=seed,
        token_accounting=_accounting(),
    )

    corrupted = objective.corrupt(
        torch.arange(batch_size * 8, dtype=torch.long).reshape(batch_size, 8)
    )

    torch.testing.assert_close(corrupted.diffusion_times, expected)
    assert torch.all(corrupted.diffusion_times >= 1.0e-3)
    assert torch.all(corrupted.diffusion_times <= 1.0)


def test_standard_objective_rng_is_shared_by_model_parallel_ranks() -> None:
    spec = SimpleNamespace(
        model=SimpleNamespace(seq_len=8),
        objective=SimpleNamespace(
            noise_schedule="loglinear",
            loss_weighting="inverse_move_chance",
            noise_schedule_epsilon=1.0e-3,
            sampling_epsilon_min=1.0e-3,
            sampling_epsilon_max=1.0,
            antithetic_sampling=True,
            bp_loss_scale=None,
        ),
    )
    first = build_standard_training_task(
        spec=spec,
        runtime=None,
        device=torch.device("cpu"),
        seed=101,
        data_parallel_seed=17,
        mask_token_id=99,
        vocab_size=100,
        token_accounting=_accounting(),
        block_size=2,
    )
    second = build_standard_training_task(
        spec=spec,
        runtime=None,
        device=torch.device("cpu"),
        seed=202,
        data_parallel_seed=17,
        mask_token_id=99,
        vocab_size=100,
        token_accounting=_accounting(),
        block_size=2,
    )
    x0 = torch.arange(16, dtype=torch.long).reshape(2, 8)

    first_batch = first.objective_runtime.corrupt(x0)
    second_batch = second.objective_runtime.corrupt(x0)

    torch.testing.assert_close(
        first_batch.diffusion_times, second_batch.diffusion_times
    )
    torch.testing.assert_close(
        first_batch.noisy_input_ids, second_batch.noisy_input_ids
    )


def test_sft_denominator_matches_global_dp_token_normalization(monkeypatch) -> None:
    spec = SimpleNamespace(
        model=SimpleNamespace(seq_len=4),
        objective=SimpleNamespace(
            noise_schedule="loglinear",
            loss_weighting="inverse_move_chance",
            noise_schedule_epsilon=1.0e-3,
            sampling_epsilon_min=1.0e-3,
            sampling_epsilon_max=1.0,
            antithetic_sampling=True,
            bp_loss_scale=None,
        ),
    )
    runtime = SimpleNamespace(
        enabled=False,
        data_parallel_size=2,
        data_parallel_group=object(),
    )
    task = build_standard_training_task(
        spec=spec,
        runtime=runtime,
        device=torch.device("cpu"),
        seed=1,
        data_parallel_seed=2,
        mask_token_id=99,
        vocab_size=100,
        token_accounting=_accounting(),
        block_size=2,
    )

    def _all_reduce(count: torch.Tensor, **_: object) -> None:
        count.add_(6)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    prepared = task.prepare(
        DataBatch(
            input_ids=torch.arange(4).reshape(1, 4),
            loss_mask=torch.tensor([[False, False, True, True]]),
        )
    )
    torch.testing.assert_close(
        torch.as_tensor(prepared.corrupted.loss_denominator),
        torch.tensor(4.0, dtype=torch.float64),
    )


def test_sft_accumulation_uses_one_global_optimizer_step_denominator(
    monkeypatch,
) -> None:
    spec = SimpleNamespace(
        model=SimpleNamespace(seq_len=4),
        objective=SimpleNamespace(
            noise_schedule="loglinear",
            loss_weighting="inverse_move_chance",
            noise_schedule_epsilon=1.0e-3,
            sampling_epsilon_min=1.0e-3,
            sampling_epsilon_max=1.0,
            antithetic_sampling=True,
            bp_loss_scale=None,
        ),
    )
    runtime = SimpleNamespace(
        enabled=False,
        data_parallel_size=2,
        data_parallel_group=object(),
    )
    task = build_standard_training_task(
        spec=spec,
        runtime=runtime,
        device=torch.device("cpu"),
        seed=1,
        data_parallel_seed=2,
        mask_token_id=99,
        vocab_size=100,
        token_accounting=_accounting(),
        block_size=2,
    )
    calls = 0

    def _all_reduce(count: torch.Tensor, **_: object) -> None:
        nonlocal calls
        calls += 1
        count.add_(8)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    batches = [
        DataBatch(
            input_ids=torch.arange(4).reshape(1, 4),
            loss_mask=torch.tensor([[False, False, True, True]]),
        ),
        DataBatch(
            input_ids=torch.arange(4, 8).reshape(1, 4),
            loss_mask=torch.tensor([[False, True, True, True]]),
        ),
    ]

    denominator = task.accumulation_loss_denominator(batches)

    assert calls == 1
    torch.testing.assert_close(
        torch.as_tensor(denominator),
        torch.tensor(13.0 / 4.0, dtype=torch.float64),
    )
    prepared = [task.prepare(batch, loss_denominator=denominator) for batch in batches]
    assert calls == 1
    for item in prepared:
        torch.testing.assert_close(
            torch.as_tensor(item.corrupted.loss_denominator),
            torch.tensor(13.0 / 4.0, dtype=torch.float64),
        )


def test_sft_denominator_covers_distinct_dp_and_ep_samples(monkeypatch) -> None:
    spec = SimpleNamespace(
        model=SimpleNamespace(seq_len=4),
        objective=SimpleNamespace(
            noise_schedule="loglinear",
            loss_weighting="inverse_move_chance",
            noise_schedule_epsilon=1.0e-3,
            sampling_epsilon_min=1.0e-3,
            sampling_epsilon_max=1.0,
            antithetic_sampling=True,
            bp_loss_scale=None,
        ),
    )
    data_group = object()
    expert_group = object()
    runtime = SimpleNamespace(
        enabled=True,
        data_parallel_size=2,
        data_parallel_group=data_group,
        expert_parallel_size=2,
        expert_parallel_group=expert_group,
    )
    task = build_standard_training_task(
        spec=spec,
        runtime=runtime,
        device=torch.device("cpu"),
        seed=1,
        data_parallel_seed=2,
        mask_token_id=99,
        vocab_size=100,
        token_accounting=_accounting(),
        block_size=2,
    )
    groups: list[object] = []

    def _all_reduce(count: torch.Tensor, *, group: object, **_: object) -> None:
        groups.append(group)
        count.add_(4 if group is data_group else 10)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    batch = DataBatch(
        input_ids=torch.arange(4).reshape(1, 4),
        loss_mask=torch.tensor([[False, False, True, True]]),
    )

    denominator = task.accumulation_loss_denominator([batch])

    assert groups == [data_group, expert_group]
    torch.testing.assert_close(
        torch.as_tensor(denominator),
        torch.tensor(4.0, dtype=torch.float64),
    )


def test_unit_loss_weighting_is_explicit_at_high_corruption() -> None:
    objective = StandardBlockDiffusionObjectiveRuntime(
        mask_token_id=99,
        block_size=2,
        seq_len=8,
        device=torch.device("cpu"),
        seed=7,
        token_accounting=_accounting(),
        loss_weighting="unit",
        sampling_epsilon_min=0.75,
        sampling_epsilon_max=1.0,
    )

    corrupted = objective.corrupt(torch.arange(8).reshape(1, 8))

    torch.testing.assert_close(
        corrupted.loss_weights,
        torch.ones_like(corrupted.diffusion_times),
    )


def test_high_corruption_does_not_implicitly_change_loss_weighting() -> None:
    objective = StandardBlockDiffusionObjectiveRuntime(
        mask_token_id=99,
        block_size=2,
        seq_len=8,
        device=torch.device("cpu"),
        seed=7,
        token_accounting=_accounting(),
        sampling_epsilon_min=0.75,
        sampling_epsilon_max=1.0,
    )

    corrupted = objective.corrupt(torch.arange(8).reshape(1, 8))

    torch.testing.assert_close(
        corrupted.loss_weights,
        corrupted.diffusion_times.reciprocal(),
    )
