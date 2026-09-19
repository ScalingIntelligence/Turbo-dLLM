# Copyright 2026 The dllm_parallel Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tokenizer integration for normalized preparation records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import torch

from dllm_parallel.data.formatting import FormattedRecord, IntegerSequence
from dllm_parallel.data.schemas import SupervisionSpec, TokenizerSpec


@dataclass(frozen=True)
class TokenizedRecord:
    tokens: IntegerSequence
    loss_mask: tuple[bool, ...] | None = None
    sample_id: Any | None = None
    group_id: Any | None = None


def load_tokenizer(spec: TokenizerSpec) -> Any | None:
    """Load the configured tokenizer, or return ``None`` when not requested."""

    if spec.model is None:
        return None
    from transformers import AutoTokenizer

    kwargs = dict(spec.kwargs)
    kwargs.update(
        {
            "revision": spec.revision,
            "trust_remote_code": bool(spec.trust_remote_code),
            "use_fast": bool(spec.use_fast),
        }
    )
    return AutoTokenizer.from_pretrained(spec.model, **kwargs)


def tokenize_record(
    record: FormattedRecord,
    *,
    tokenizer: Any | None,
    tokenizer_spec: TokenizerSpec,
    supervision: SupervisionSpec,
) -> TokenizedRecord:
    """Tokenize one normalized record and construct its exact loss mask."""

    tokens, loss_mask = _tokenize_content(
        record,
        tokenizer=tokenizer,
        tokenizer_spec=tokenizer_spec,
        policy=supervision.policy,
    )
    tokens, loss_mask = _append_eos_if_requested(
        tokens,
        loss_mask,
        tokenizer=tokenizer,
        tokenizer_spec=tokenizer_spec,
    )
    _validate_tokenized_record(tokens, loss_mask, tokenizer)
    return TokenizedRecord(
        tokens=tokens,
        loss_mask=loss_mask,
        sample_id=record.sample_id,
        group_id=record.group_id,
    )


def _tokenize_content(
    record: FormattedRecord,
    *,
    tokenizer: Any | None,
    tokenizer_spec: TokenizerSpec,
    policy: str,
) -> tuple[IntegerSequence, tuple[bool, ...] | None]:
    if record.kind == "pretokenized":
        return _tokens(record.input_ids), _pretokenized_mask(record, policy)
    if record.kind == "text":
        _require_tokenizer(tokenizer, record.kind)
        return tuple(_encode(tokenizer, record.text or "")), None
    if record.kind == "prompt_completion":
        return _tokenize_prompt_completion(
            record,
            tokenizer=tokenizer,
            tokenizer_spec=tokenizer_spec,
            policy=policy,
        )
    if record.kind == "messages":
        _require_tokenizer(tokenizer, record.kind)
        tokens, assistant_mask = _chat_tokens(
            tokenizer,
            list(record.messages or ()),
            tokenizer_spec,
            require_assistant_mask=policy == "assistant_only",
        )
        return tokens, assistant_mask if policy == "assistant_only" else None
    raise ValueError(f"unsupported formatted record kind: {record.kind!r}")


def _tokenize_prompt_completion(
    record: FormattedRecord,
    *,
    tokenizer: Any | None,
    tokenizer_spec: TokenizerSpec,
    policy: str,
) -> tuple[tuple[int, ...], tuple[bool, ...] | None]:
    _require_tokenizer(tokenizer, record.kind)
    if record.prompt_messages is not None:
        prompt_tokens, _ = _chat_tokens(
            tokenizer,
            list(record.prompt_messages),
            tokenizer_spec,
            require_assistant_mask=False,
            add_generation_prompt=True,
        )
        tokens, _ = _chat_tokens(
            tokenizer,
            list(record.prompt_messages + (record.completion_messages or ())),
            tokenizer_spec,
            require_assistant_mask=False,
            add_generation_prompt=False,
        )
        if tokens[: len(prompt_tokens)] != prompt_tokens:
            raise RuntimeError(
                "the chat template does not preserve the prompt as a stable token "
                "prefix"
            )
        completion_length = len(tokens) - len(prompt_tokens)
    else:
        prompt_tokens = tuple(_encode(tokenizer, record.prompt or ""))
        completion_tokens = tuple(_encode(tokenizer, record.completion or ""))
        tokens = prompt_tokens + completion_tokens
        completion_length = len(completion_tokens)
    loss_mask = (
        tuple([False] * len(prompt_tokens) + [True] * completion_length)
        if policy == "completion_only"
        else None
    )
    return tokens, loss_mask


def _append_eos_if_requested(
    tokens: IntegerSequence,
    loss_mask: tuple[bool, ...] | None,
    *,
    tokenizer: Any | None,
    tokenizer_spec: TokenizerSpec,
) -> tuple[IntegerSequence, tuple[bool, ...] | None]:
    if len(tokens) == 0:
        raise ValueError("tokenized record contains no tokens")
    if tokenizer_spec.add_eos:
        _require_tokenizer(tokenizer, "EOS insertion")
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("tokenizer has no eos_token_id")
        if tokens[-1] != int(eos_token_id):
            tokens = _append_token(tokens, int(eos_token_id))
            if loss_mask is not None:
                loss_mask = (*loss_mask, bool(loss_mask[-1]))
    return tokens, loss_mask


def _validate_tokenized_record(
    tokens: IntegerSequence,
    loss_mask: tuple[bool, ...] | None,
    tokenizer: Any | None,
) -> None:
    if loss_mask is not None:
        if len(loss_mask) != len(tokens):
            raise ValueError("loss mask must have the same length as input_ids")
        if not any(loss_mask):
            raise ValueError("tokenized record contains no supervised tokens")
    mask_token_id = (
        None if tokenizer is None else getattr(tokenizer, "mask_token_id", None)
    )
    if mask_token_id is not None and _contains_token(tokens, int(mask_token_id)):
        raise ValueError("source record contains the reserved mask token")
    if _minimum_token(tokens) < 0:
        raise ValueError("token IDs must be nonnegative")


def _pretokenized_mask(
    record: FormattedRecord,
    policy: str,
) -> tuple[bool, ...] | None:
    if policy == "full":
        return None
    if policy == "assistant_only":
        value = record.assistant_mask
    elif policy == "completion_only":
        value = record.completion_mask
    else:
        value = record.loss_mask
        if value is None and record.labels is not None:
            value = tuple(label != -100 for label in record.labels)
    if value is None:
        raise ValueError(f"pretokenized record does not provide a {policy} loss mask")
    return tuple(bool(item) for item in value)


def _chat_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    spec: TokenizerSpec,
    *,
    require_assistant_mask: bool,
    add_generation_prompt: bool = False,
) -> tuple[tuple[int, ...], tuple[bool, ...] | None]:
    kwargs = dict(spec.chat_template_kwargs)
    reserved = {
        "add_generation_prompt",
        "chat_template",
        "return_assistant_tokens_mask",
        "return_dict",
        "tokenize",
    }
    overlap = sorted(reserved.intersection(kwargs))
    if overlap:
        raise ValueError(
            "tokenizer.chat_template_kwargs may not override: " + ", ".join(overlap)
        )
    kwargs.update(
        {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
    )
    if spec.chat_template is not None:
        kwargs["chat_template"] = spec.chat_template
    if require_assistant_mask:
        kwargs.update({"return_dict": True, "return_assistant_tokens_mask": True})
    try:
        payload = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if require_assistant_mask:
            raise RuntimeError(
                "assistant-only supervision requires a chat template capable of "
                "returning an assistant token mask"
            ) from exc
        raise
    if isinstance(payload, Mapping):
        ids = _flatten_ids(payload.get("input_ids"), "chat input_ids")
        mask_value = payload.get("assistant_masks", payload.get("assistant_mask"))
        mask = (
            None
            if mask_value is None
            else tuple(
                bool(item) for item in _flatten_ids(mask_value, "assistant mask")
            )
        )
    else:
        ids = _flatten_ids(payload, "chat input_ids")
        mask = None
    if require_assistant_mask and mask is None:
        raise RuntimeError(
            "assistant-only supervision requires the chat template to return an "
            "assistant token mask"
        )
    if mask is not None and len(mask) != len(ids):
        raise ValueError("assistant token mask must match chat input_ids")
    return tuple(ids), mask


def _flatten_ids(value: Any, name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise ValueError(f"{name} must contain one record")
        value = value[0]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a token sequence")
    return [int(item) for item in value]


def _tokens(value: IntegerSequence | None) -> IntegerSequence:
    if value is None:
        raise ValueError("pretokenized record is missing input_ids")
    return value


def _append_token(tokens: IntegerSequence, token: int) -> IntegerSequence:
    if isinstance(tokens, tuple):
        return (*tokens, token)
    return _SuffixedIntegerSequence(tokens, token)


def _contains_token(tokens: IntegerSequence, token: int) -> bool:
    if isinstance(tokens, np.ndarray):
        return bool(np.any(tokens == token))
    if isinstance(tokens, torch.Tensor):
        return bool(torch.any(tokens == token).item())
    return token in tokens


def _minimum_token(tokens: IntegerSequence) -> int:
    if isinstance(tokens, np.ndarray):
        return int(tokens.min())
    if isinstance(tokens, torch.Tensor):
        return int(tokens.min().item())
    return min(tokens)


@dataclass(frozen=True)
class _SuffixedIntegerSequence(Sequence[int]):
    values: IntegerSequence
    token: int

    def __len__(self) -> int:
        return len(self.values) + 1

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return tuple(self[position] for position in range(start, stop, step))
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        if normalized == len(self.values):
            return self.token
        return int(self.values[normalized])


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(item) for item in tokenizer.encode(text, add_special_tokens=False)]


def _require_tokenizer(tokenizer: Any | None, kind: str) -> None:
    if tokenizer is None:
        raise RuntimeError(f"{kind} preparation requires tokenizer.model")


__all__ = ("TokenizedRecord", "load_tokenizer", "tokenize_record")
