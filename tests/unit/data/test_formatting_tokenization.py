from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from dllm_parallel.data.formatting import FormattedRecord, format_record
from dllm_parallel.data.registry import register_formatter
from dllm_parallel.data.schemas import RecordSpec, SupervisionSpec, TokenizerSpec
from dllm_parallel.data.tokenization import TokenizedRecord, tokenize_record


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    mask_token_id = 98
    chat_template = "{% generation %}assistant{% endgeneration %}"

    def __len__(self) -> int:
        return 100

    def get_vocab(self) -> dict[str, int]:
        return {str(index): index for index in range(100)}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) % 50 + 1 for character in text]

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        assert kwargs["tokenize"] is True
        ids: list[int] = []
        mask: list[int] = []
        for message in messages:
            content = self.encode(str(message["content"]), add_special_tokens=False)
            ids.extend(content)
            mask.extend([int(message["role"] == "assistant")] * len(content))
        if kwargs.get("return_dict"):
            return {"input_ids": ids, "assistant_masks": mask}
        return ids


def test_format_record_supports_text_messages_prompt_completion_and_ids() -> None:
    assert format_record("raw", RecordSpec(type="text")) == FormattedRecord(
        kind="text", text="raw"
    )
    assert (
        format_record(
            {"body": "mapped"}, RecordSpec(type="text", text_field="body")
        ).text
        == "mapped"
    )

    messages = format_record(
        {
            "conversation": [
                {"speaker": "user", "value": "question", "metadata": 1},
                {"speaker": "assistant", "value": "answer"},
            ]
        },
        RecordSpec(
            type="messages",
            messages_field="conversation",
            role_field="speaker",
            content_field="value",
        ),
    )
    assert messages.messages == (
        {"role": "user", "content": "question", "metadata": 1},
        {"role": "assistant", "content": "answer"},
    )

    pair = format_record(
        {"request": "p", "response": "c", "id": 7, "trajectory": 3},
        RecordSpec(
            type="prompt_completion",
            prompt_field="request",
            completion_field="response",
            sample_id_field="id",
            group_id_field="trajectory",
        ),
    )
    assert (pair.prompt, pair.completion, pair.sample_id, pair.group_id) == (
        "p",
        "c",
        7,
        3,
    )

    conversational_pair = format_record(
        {
            "prompt": [{"role": "user", "content": "question"}],
            "completion": [
                {"role": "assistant", "content": "answer", "tool_calls": [1]}
            ],
        },
        RecordSpec(type="prompt_completion"),
    )
    assert conversational_pair.prompt_messages == (
        {"role": "user", "content": "question"},
    )
    assert conversational_pair.completion_messages == (
        {"role": "assistant", "content": "answer", "tool_calls": [1]},
    )

    tokenized = format_record(
        {"nested": {"ids": [1, 2]}, "mask": [0, 1]},
        RecordSpec(
            type="pretokenized",
            input_ids_field="nested.ids",
            loss_mask_field="mask",
        ),
    )
    assert tokenized.input_ids == (1, 2)
    assert tokenized.loss_mask == (False, True)


def test_format_record_reports_missing_or_invalid_fields() -> None:
    with pytest.raises(ValueError, match="text field"):
        format_record({}, RecordSpec(type="text"))
    with pytest.raises(ValueError, match=r"messages.*sequence"):
        format_record({"messages": "not-a-list"}, RecordSpec(type="messages"))
    with pytest.raises(ValueError, match=r"role.*content"):
        format_record({"messages": [{"role": "user"}]}, RecordSpec(type="messages"))
    with pytest.raises(ValueError, match="both strings or both message sequences"):
        format_record(
            {
                "prompt": "question",
                "completion": [{"role": "assistant", "content": "answer"}],
            },
            RecordSpec(type="prompt_completion"),
        )


def test_custom_formatter_must_return_formatted_record() -> None:
    register_formatter(
        "unit-custom-record",
        lambda record, spec: FormattedRecord(kind="text", text=record["value"]),
        replace=True,
    )
    assert (
        format_record({"value": "custom"}, RecordSpec(type="unit-custom-record")).text
        == "custom"
    )

    register_formatter("unit-invalid-record", lambda record, spec: record, replace=True)
    with pytest.raises(TypeError, match="FormattedRecord"):
        format_record({}, RecordSpec(type="unit-invalid-record"))


def test_tokenize_text_and_prompt_completion_with_eos() -> None:
    tokenizer = FakeTokenizer()
    text = tokenize_record(
        FormattedRecord(kind="text", text="ab"),
        tokenizer=tokenizer,
        tokenizer_spec=TokenizerSpec(add_eos=True),
        supervision=SupervisionSpec(policy="full"),
    )
    assert text == TokenizedRecord(tokens=(48, 49, 99))

    pair = tokenize_record(
        FormattedRecord(kind="prompt_completion", prompt="a", completion="b"),
        tokenizer=tokenizer,
        tokenizer_spec=TokenizerSpec(add_eos=True),
        supervision=SupervisionSpec(policy="completion_only"),
    )
    assert pair.tokens == (48, 49, 99)
    assert pair.loss_mask == (False, True, True)


def test_tokenize_conversational_prompt_completion_uses_verified_prefix() -> None:
    class ConversationTokenizer(FakeTokenizer):
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> Any:
            ids: list[int] = []
            for message in messages:
                if message["role"] == "assistant":
                    ids.append(77)
                ids.extend(self.encode(message["content"], add_special_tokens=False))
            if kwargs.get("add_generation_prompt"):
                ids.append(77)
            return ids

    tokenizer = ConversationTokenizer()
    record = FormattedRecord(
        kind="prompt_completion",
        prompt_messages=({"role": "user", "content": "a"},),
        completion_messages=({"role": "assistant", "content": "bc"},),
    )
    result = tokenize_record(
        record,
        tokenizer=tokenizer,
        tokenizer_spec=TokenizerSpec(),
        supervision=SupervisionSpec(policy="completion_only"),
    )

    assert result.tokens == (48, 77, 49, 50)
    assert result.loss_mask == (False, False, True, True)

    class NonPrefixTokenizer(ConversationTokenizer):
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> Any:
            return [1, 2] if kwargs.get("add_generation_prompt") else [1, 3, 4]

    with pytest.raises(RuntimeError, match="stable token prefix"):
        tokenize_record(
            record,
            tokenizer=NonPrefixTokenizer(),
            tokenizer_spec=TokenizerSpec(),
            supervision=SupervisionSpec(policy="completion_only"),
        )


def test_tokenize_messages_requires_and_uses_template_assistant_mask() -> None:
    tokenizer = FakeTokenizer()
    record = FormattedRecord(
        kind="messages",
        messages=(
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "bc"},
        ),
    )
    result = tokenize_record(
        record,
        tokenizer=tokenizer,
        tokenizer_spec=TokenizerSpec(add_eos=False),
        supervision=SupervisionSpec(policy="assistant_only"),
    )
    assert result.tokens == (48, 49, 50)
    assert result.loss_mask == (False, True, True)

    class NoMaskTokenizer(FakeTokenizer):
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> Any:
            return {"input_ids": [1, 2]}

    with pytest.raises(RuntimeError, match="assistant token mask"):
        tokenize_record(
            record,
            tokenizer=NoMaskTokenizer(),
            tokenizer_spec=TokenizerSpec(add_eos=False),
            supervision=SupervisionSpec(policy="assistant_only"),
        )


def test_pretokenized_masks_and_reserved_mask_token_validation() -> None:
    provided = tokenize_record(
        FormattedRecord(
            kind="pretokenized",
            input_ids=(1, 2, 3),
            labels=(-100, 2, 3),
        ),
        tokenizer=None,
        tokenizer_spec=TokenizerSpec(add_eos=False),
        supervision=SupervisionSpec(policy="provided"),
    )
    assert provided.loss_mask == (False, True, True)

    tensor_record = format_record(
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "loss_mask": np.asarray([0, 1, 1], dtype=np.uint8),
        },
        RecordSpec(type="pretokenized"),
    )
    assert torch.equal(tensor_record.input_ids, torch.tensor([1, 2, 3]))
    assert tensor_record.loss_mask == (False, True, True)

    with pytest.raises(ValueError, match="reserved mask token"):
        tokenize_record(
            FormattedRecord(kind="pretokenized", input_ids=(1, 98, 3)),
            tokenizer=FakeTokenizer(),
            tokenizer_spec=TokenizerSpec(add_eos=False),
            supervision=SupervisionSpec(policy="full"),
        )


def test_pretokenized_integer_arrays_remain_zero_copy_until_packing() -> None:
    input_ids = np.arange(1024, dtype=np.int32)

    formatted = format_record(
        {"input_ids": input_ids},
        RecordSpec(type="pretokenized"),
    )
    tokenized = tokenize_record(
        formatted,
        tokenizer=None,
        tokenizer_spec=TokenizerSpec(add_eos=False),
        supervision=SupervisionSpec(policy="full"),
    )

    assert formatted.input_ids is input_ids
    assert tokenized.tokens is input_ids


def test_tokenization_rejects_mask_length_and_empty_supervision() -> None:
    with pytest.raises(ValueError, match="same length"):
        tokenize_record(
            FormattedRecord(kind="pretokenized", input_ids=(1, 2), loss_mask=(True,)),
            tokenizer=None,
            tokenizer_spec=TokenizerSpec(add_eos=False),
            supervision=SupervisionSpec(policy="provided"),
        )


def test_added_eos_inherits_only_the_final_token_supervision() -> None:
    result = tokenize_record(
        FormattedRecord(
            kind="pretokenized",
            input_ids=(1, 2, 3),
            loss_mask=(True, False, False),
        ),
        tokenizer=FakeTokenizer(),
        tokenizer_spec=TokenizerSpec(add_eos=True),
        supervision=SupervisionSpec(policy="provided"),
    )

    assert result.tokens == (1, 2, 3, 99)
    assert result.loss_mask == (True, False, False, False)
    with pytest.raises(ValueError, match="no supervised tokens"):
        tokenize_record(
            FormattedRecord(
                kind="pretokenized", input_ids=(1, 2), loss_mask=(False, False)
            ),
            tokenizer=None,
            tokenizer_spec=TokenizerSpec(add_eos=False),
            supervision=SupervisionSpec(policy="provided"),
        )
