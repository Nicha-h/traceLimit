from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch

from config import DEPTHS, MODELS, SINGLE_GPU_SAFE_DEPTH_LIMIT

_HF_IDS = {name: cfg["hf_id"] for name, cfg in MODELS.items()}
_GPU_REQS = {name: cfg["gpu_requirement"] for name, cfg in MODELS.items()}

# Only DeepSeek-Coder-V2-Lite requires trust_remote_code due to custom MLA kernels.
_TRUST_REMOTE_CODE = {"deepseek-coder-v2-lite-instruct"}


def check_gpu_for_model(model_key: str) -> None:
    """Print a warning before loading a dual_t4 model on a single-GPU session."""
    n_gpus = torch.cuda.device_count()
    if _GPU_REQS.get(model_key) == "dual_t4" and n_gpus < 2:
        print(
            f"WARNING [{model_key}]: requires dual-T4 (2×16 GB) for full 128K context "
            f"but only {n_gpus} GPU(s) detected. "
            f"Depths >{SINGLE_GPU_SAFE_DEPTH_LIMIT:.0%} will be skipped to avoid OOM "
            f"(GQA KV cache + weights exceed ~14 GB above that point)."
        )


def is_depth_safe(model_key: str, depth: float) -> bool:
    """Return False if running this depth on the current GPU setup would risk OOM."""
    if _GPU_REQS.get(model_key) == "dual_t4" and torch.cuda.device_count() < 2:
        safe = depth <= SINGLE_GPU_SAFE_DEPTH_LIMIT
        if not safe:
            print(
                f"INFO [{model_key}]: skipping depth {depth:.2f} — single-GPU session, "
                f"GQA KV cache + weights would exceed ~14 GB headroom."
            )
        return safe
    return True


@lru_cache(maxsize=None)
def _load_transformers_runtime(model_key: str) -> dict[str, Any]:
    import importlib

    check_gpu_for_model(model_key)

    transformers = importlib.import_module("transformers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer
    BitsAndBytesConfig = transformers.BitsAndBytesConfig

    model_id = _HF_IDS[model_key]
    trust_remote = model_key in _TRUST_REMOTE_CODE
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if trust_remote:
        import transformers.utils.import_utils as _tui
        if not hasattr(_tui, "is_torch_fx_available"):
            _tui.is_torch_fx_available = lambda: False
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=trust_remote, token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=trust_remote,
        token=hf_token,
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

    if isinstance(prompt, list):
        messages = prompt  # already a list of dicts from build_prompt
    else:
        messages = [{"role": "user", "content": prompt}]  # backwards compat
    _tok_out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    # Newer transformers returns BatchEncoding; older returns a raw tensor.
    if hasattr(_tok_out, "input_ids"):
        input_ids = _tok_out.input_ids.to(model.device)
    else:
        input_ids = _tok_out.to(model.device)

    # Explicit mask prevents the "pad==eos, cannot infer mask" warning.
    attention_mask = torch.ones_like(input_ids)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"[OOM] input length {input_ids.shape[-1]} tokens — skipping this call")
        return ""

    generated = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)
