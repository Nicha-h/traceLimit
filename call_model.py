from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

from config import MODELS


MODEL_LOADERS = {model_name: model_config["model_id"] for model_name, model_config in MODELS.items()}


@lru_cache(maxsize=None)
def _load_transformers_runtime(model_key: str) -> dict[str, Any]:
    transformers = importlib.import_module("transformers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer
    BitsAndBytesConfig = transformers.BitsAndBytesConfig

    model_id = MODEL_LOADERS[model_key]
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


def _call_local_model(model_bundle: dict[str, Any], prompt: str, temperature: float = 0.0) -> str:
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    max_new_tokens = model_bundle.get("max_new_tokens", 4096)

    messages = [
        {"role": "system", "content": "You are an expert Python debugger."},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
    )

    device = model_bundle.get("device")
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = None
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0.0,
    }
    if getattr(tokenizer, "pad_token_id", None) is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    output_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def get_local_runtime(model_key: str) -> dict[str, Any]:
    return _load_transformers_runtime(model_key)


def call_model(runtime: dict, prompt: str, temperature: float = 0.0) -> str:
    """
    Executes a completion call against a locally loaded Hugging Face model bundle.
    """
    if not isinstance(runtime, dict) or "model" not in runtime or "tokenizer" not in runtime:
        raise TypeError("call_model expects a runtime dictionary with 'model' and 'tokenizer' entries")
    return _call_local_model(runtime, prompt, temperature=temperature)