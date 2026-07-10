from __future__ import annotations

from typing import Any

import openai


def _call_api_model(model_config: dict, prompt: str, temperature: float = 0.0) -> str:
    client = openai.OpenAI(
        base_url=model_config["api_base"],
        api_key=model_config["api_key"],
    )

    messages = [
        {"role": "system", "content": "You are an expert Python debugger."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model_config["model_id"],
        messages=messages,
        temperature=temperature,
        max_tokens=model_config.get("max_tokens", 4096),
    )
    return response.choices[0].message.content or ""


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


def call_model(model_config: dict, prompt: str, temperature: float = 0.0) -> str:
    """
    Executes a completion call against either an API-backed model config or a locally loaded Hugging Face model bundle.
    """
    if isinstance(model_config, dict) and "model" in model_config and "tokenizer" in model_config:
        return _call_local_model(model_config, prompt, temperature=temperature)
    return _call_api_model(model_config, prompt, temperature=temperature)