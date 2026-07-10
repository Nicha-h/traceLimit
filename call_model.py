from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from config import MODELS

_HF_IDS = {name: cfg["hf_id"] for name, cfg in MODELS.items()}


@lru_cache(maxsize=None)
def _load_transformers_runtime(model_key: str) -> dict[str, Any]:
    import importlib

    transformers = importlib.import_module("transformers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer
    BitsAndBytesConfig = transformers.BitsAndBytesConfig

    model_id = _HF_IDS[model_key]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return {"model": model, "tokenizer": tokenizer, "max_new_tokens": 4096}


def get_local_runtime(model_key: str) -> dict[str, Any]:
    return _load_transformers_runtime(model_key)


def call_model(runtime: dict, prompt: str, temperature: float = 0.0) -> str:
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    max_new_tokens = runtime.get("max_new_tokens", 4096)

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)
