from __future__ import annotations

import pytest

from dllm_parallel.data.packing import (
    pack_full_sequence_records,
    prepare_indexed_record,
)
from dllm_parallel.data.schemas import PackingSpec
from dllm_parallel.data.tokenization import TokenizedRecord


def test_prepare_indexed_record_truncates_overflow_and_alignment_explicitly() -> None:
    result = prepare_indexed_record(
        TokenizedRecord(
            tokens=tuple(range(11)),
            loss_mask=(False,) * 5 + (True,) * 6,
            sample_id=4,
        ),
        PackingSpec(
            maximum_length=8,
            alignment=4,
            overflow="truncate_left",
            alignment_policy="truncate_left",
        ),
    )

    assert result.record.tokens == tuple(range(3, 11))
    assert result.record.loss_mask == (False, False, True, True, True, True, True, True)
    assert result.truncated_tokens == 3
    assert result.padding_tokens == 0


def test_prepare_indexed_record_can_align_by_truncating_or_padding() -> None:
    record = TokenizedRecord(
        tokens=(1, 2, 3, 4, 5, 6),
        loss_mask=(False, False, True, True, True, True),
    )
    truncated = prepare_indexed_record(
        record,
        PackingSpec(
            maximum_length=8,
            alignment=4,
            alignment_policy="truncate_right",
        ),
    )
    padded = prepare_indexed_record(
        record,
        PackingSpec(
            maximum_length=8,
            alignment=4,
            alignment_policy="pad_left",
            pad_token_id=0,
        ),
    )

    assert truncated.record.tokens == (1, 2, 3, 4)
    assert truncated.record.loss_mask == (False, False, True, True)
    assert truncated.truncated_tokens == 2
    assert padded.record.tokens == (0, 0, 1, 2, 3, 4, 5, 6)
    assert padded.record.loss_mask == (
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    )
    assert padded.padding_tokens == 2


def test_prepare_indexed_record_rejects_implicit_or_invalid_transformations() -> None:
    record = TokenizedRecord(tokens=(1, 2, 3), loss_mask=(False, True, True))
    with pytest.raises(ValueError, match="exceeds maximum"):
        prepare_indexed_record(
            record,
            PackingSpec(maximum_length=2, overflow="reject"),
        )
    with pytest.raises(ValueError, match="not aligned"):
        prepare_indexed_record(
            record,
            PackingSpec(
                maximum_length=4,
                alignment=4,
                alignment_policy="reject",
            ),
        )
    with pytest.raises(ValueError, match="pad_token_id"):
        prepare_indexed_record(
            record,
            PackingSpec(
                maximum_length=4,
                alignment=4,
                alignment_policy="pad_right",
            ),
        )
    with pytest.raises(ValueError, match="no supervised tokens"):
        prepare_indexed_record(
            TokenizedRecord(
                tokens=(1, 2, 3),
                loss_mask=(True, False, False),
            ),
            PackingSpec(
                maximum_length=4,
                alignment=2,
                alignment_policy="truncate_left",
                minimum_length=2,
            ),
        )


def test_pack_full_sequence_records_preserves_document_boundaries_with_separator() -> (
    None
):
    result = pack_full_sequence_records(
        [
            TokenizedRecord(tokens=(1, 2, 3)),
            TokenizedRecord(tokens=(4, 5)),
        ],
        PackingSpec(
            maximum_length=2,
            overflow="truncate_right",
            separator_token_id=9,
        ),
    )

    assert result.tokens == (1, 2, 9, 4, 5)
    assert result.record_count == 2
    assert result.truncated_tokens == 1
    assert result.separator_tokens == 1


def test_pack_full_sequence_rejects_supervised_records_and_empty_output() -> None:
    spec = PackingSpec(maximum_length=4)
    with pytest.raises(ValueError, match="supervision mask"):
        pack_full_sequence_records(
            [TokenizedRecord(tokens=(1,), loss_mask=(True,))], spec
        )
    with pytest.raises(ValueError, match="no tokens"):
        pack_full_sequence_records([], spec)


def test_pack_full_sequence_enforces_minimum_length_like_artifact_writer() -> None:
    with pytest.raises(ValueError, match="below configured minimum"):
        pack_full_sequence_records(
            [TokenizedRecord(tokens=(1, 2))],
            PackingSpec(maximum_length=4, minimum_length=3),
        )
