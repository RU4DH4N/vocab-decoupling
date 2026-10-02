import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from data.tokenizers import train_bpe
from execution.external import generate, logits, special_ids, windowed_nats

CONFIG = {
    "precision": "auto",
    "generation": {"symbols": 40, "bytes": 40, "temperature": 1.0, "top_p": 1.0},
}


def tiny(context):
    tokenizer = train_bpe(300, ["the cat sat on the mat"] * 4, use_regex=True)
    tokenizer.add_special_tokens(["</s>"])
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=context,
        )
    ).eval()
    return model, tokenizer


def test_generation_never_samples_special_tokens():
    model, tokenizer = tiny(64)
    stop = special_ids(tokenizer)
    assert stop == [tokenizer.token_to_id("</s>")]
    with torch.no_grad():
        model.lm_head.weight[stop] = 0
        model.lm_head.weight[stop, 0] = 1e4
    record = generate(model, tokenizer, 64, "the cat", CONFIG, 0)
    assert record["stop_reason"] == "byte-budget"
    assert "</s>" not in record["text"]


def test_generation_slides_past_the_model_context():
    model, tokenizer = tiny(4)
    record = generate(model, tokenizer, 4, "the", CONFIG, 0)
    assert record["stop_reason"] == "byte-budget"
    long = generate(model, tokenizer, 4, "the cat sat on the mat", CONFIG, 0)
    assert long["prompt_truncated"] and long["stop_reason"] == "byte-budget"
    assert not record["prompt_truncated"]


@pytest.mark.parametrize("context", [4, 5, 64])
def test_windowed_scoring_keeps_half_the_context(context):
    model, _ = tiny(64)
    torch.manual_seed(1)
    ids = torch.randint(0, model.config.vocab_size, (2, 11))
    targets = torch.randint(0, model.config.vocab_size, (2, 11))
    targets[1, 7:] = -100
    with torch.no_grad():
        nats = windowed_nats(model, ids, targets, CONFIG, context)
        kept = context - context // 2
        for position in range(ids.shape[1]):
            start = 0
            while position >= start + context:
                start += context - kept
            scores, _ = logits(model, ids[:, start : position + 1], CONFIG)
            expected = torch.nn.functional.cross_entropy(
                scores[:, -1], targets[:, position], reduction="none"
            )
            torch.testing.assert_close(
                nats[:, position].float(), expected, atol=1e-5, rtol=1e-5
            )
