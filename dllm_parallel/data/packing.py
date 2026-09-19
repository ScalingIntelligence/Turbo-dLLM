# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Explicit offline truncation, alignment, and stream packing policies."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace

from dllm_parallel.data.schemas import PackingSpec
from dllm_parallel.data.formatting import IntegerSequence
from dllm_parallel.data.tokenization import TokenizedRecord


@dataclass(frozen=True)
class IndexedPackingResult:
    record: TokenizedRecord
    truncated_tokens: int = 0
    padding_tokens: int = 0


@dataclass(frozen=True)
class PackedTokenStream:
    tokens: tuple[int, ...]
    record_count: int
    source_tokens: int
    truncated_tokens: int
    separator_tokens: int


@dataclass
class PackedTokenAccounting:
    record_count: int = 0
    logical_chunks: int = 0
    source_tokens: int = 0
    stored_tokens: int = 0
    truncated_tokens: int = 0
    separator_tokens: int = 0
    split_records: int = 0


def prepare_indexed_record(
    record: TokenizedRecord,
    spec: PackingSpec,
) -> IndexedPackingResult:
    """Apply configured length policies to one record-level sample."""

    tokens = record.tokens
    mask = record.loss_mask
    if mask is None:
        mask = (True,) * len(tokens)
    if len(mask) != len(tokens):
        raise ValueError("loss mask must have the same length as tokens")
    truncated = 0
    padding = 0
    if len(tokens) > spec.maximum_length:
        amount = len(tokens) - spec.maximum_length
        if spec.overflow == "reject":
            raise ValueError("record exceeds maximum length")
        tokens, mask = _truncate(tokens, mask, amount, side=spec.overflow)
        truncated += amount
    remainder = len(tokens) % spec.alignment
    if remainder:
        policy = spec.alignment_policy
        if policy == "reject":
            raise ValueError("record length is not aligned")
        if policy.startswith("truncate_"):
            tokens, mask = _truncate(tokens, mask, remainder, side=policy)
            truncated += remainder
        else:
            amount = spec.alignment - remainder
            if spec.pad_token_id is None:
                raise ValueError("alignment padding requires packing.pad_token_id")
            pad_tokens = (int(spec.pad_token_id),) * amount
            pad_mask = (False,) * amount
            if policy == "pad_left":
                tokens, mask = pad_tokens + tuple(tokens), pad_mask + mask
            else:
                tokens, mask = tuple(tokens) + pad_tokens, mask + pad_mask
            padding += amount
    minimum = spec.alignment if spec.minimum_length is None else spec.minimum_length
    if len(tokens) < minimum:
        raise ValueError(
            f"record length {len(tokens)} is below configured minimum {minimum}"
        )
    if not any(mask):
        raise ValueError("record contains no supervised tokens after packing")
    return IndexedPackingResult(
        record=replace(record, tokens=tuple(tokens), loss_mask=tuple(mask)),
        truncated_tokens=truncated,
        padding_tokens=padding,
    )


def pack_full_sequence_records(
    records: Iterable[TokenizedRecord],
    spec: PackingSpec,
) -> PackedTokenStream:
    """Concatenate full-sequence documents into one efficient token stream."""

    output: list[int] = []
    accounting = PackedTokenAccounting()
    for segment in iter_packed_segments(records, spec, accounting=accounting):
        output.extend(int(token) for token in segment)
    if not output:
        raise ValueError("prepared packed dataset contains no tokens")
    return PackedTokenStream(
        tokens=tuple(output),
        record_count=accounting.record_count,
        source_tokens=accounting.source_tokens,
        truncated_tokens=accounting.truncated_tokens,
        separator_tokens=accounting.separator_tokens,
    )


def iter_packed_segments(
    records: Iterable[TokenizedRecord],
    spec: PackingSpec,
    *,
    accounting: PackedTokenAccounting,
) -> Iterator[Sequence[int]]:
    """Yield bounded segments using the package's single packed-data policy."""

    for record in records:
        if record.loss_mask is not None:
            raise ValueError("packed token streams cannot contain a supervision mask")
        tokens = record.tokens
        accounting.source_tokens += len(tokens)
        split = False
        if len(tokens) > spec.maximum_length:
            amount = len(tokens) - spec.maximum_length
            if spec.overflow == "reject":
                raise ValueError("record exceeds maximum length")
            if spec.overflow == "split":
                accounting.split_records += 1
                split = True
            else:
                tokens = _truncate_tokens(tokens, amount, side=spec.overflow)
                accounting.truncated_tokens += amount
        minimum = spec.minimum_length or 1
        if len(tokens) < minimum:
            raise ValueError(
                f"record length {len(tokens)} is below configured minimum {minimum}"
            )
        if accounting.record_count and spec.separator_token_id is not None:
            separator = (int(spec.separator_token_id),)
            accounting.separator_tokens += 1
            accounting.stored_tokens += 1
            yield separator
        if split:
            starts = range(0, len(tokens), spec.maximum_length)
        else:
            starts = range(0, 1)
        for start in starts:
            stop = (
                min(start + spec.maximum_length, len(tokens)) if split else len(tokens)
            )
            segment = tokens[start:stop]
            accounting.logical_chunks += 1
            accounting.stored_tokens += len(segment)
            yield segment
        accounting.record_count += 1


def _truncate(
    tokens: IntegerSequence,
    mask: tuple[bool, ...],
    amount: int,
    *,
    side: str,
) -> tuple[IntegerSequence, tuple[bool, ...]]:
    if amount <= 0:
        return tokens, mask
    if amount >= len(tokens):
        return (), ()
    if side == "truncate_left":
        return tokens[amount:], mask[amount:]
    if side == "truncate_right":
        return tokens[:-amount], mask[:-amount]
    raise ValueError(f"unsupported truncation side: {side}")


def _truncate_tokens(
    tokens: IntegerSequence,
    amount: int,
    *,
    side: str,
) -> IntegerSequence:
    if side == "truncate_left":
        return tokens[amount:]
    if side == "truncate_right":
        return tokens[:-amount]
    raise ValueError(f"unsupported truncation side: {side}")


__all__ = (
    "IndexedPackingResult",
    "PackedTokenStream",
    "pack_full_sequence_records",
    "prepare_indexed_record",
)
