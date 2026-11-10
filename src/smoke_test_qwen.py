#!/usr/bin/env python3
"""Offline GPU smoke test for the pinned Qwen3-8B-AWQ checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-8B-AWQ"
MODEL_REVISION = "cb7d6a337aadb4d2082ed0dcef1032e4f8645194"
SEED = 42
NON_THINKING_ASSISTANT_PREFILL = "<think>\n\n</think>\n\n"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def mib(value: int) -> float:
    return round(value / (1024 * 1024), 3)


def running_gpu_compute_processes() -> list[str]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve()
    required_files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing_files = [name for name in required_files if not (model_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing model files: {missing_files}")
    existing_processes = running_gpu_compute_processes()
    if existing_processes:
        raise RuntimeError(
            "GPU safety check failed; existing compute processes: "
            + "; ".join(existing_processes)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device_index = 0
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)

    print("gpu_safety_check=PASS", flush=True)
    print(f"model_dir={model_dir}", flush=True)
    print("loading_tokenizer...", flush=True)
    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer_seconds = time.perf_counter() - tokenizer_start

    print("loading_model...", flush=True)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        # AutoAWQ's CUDA GEMM extension expects FP16 activations.  Using
        # "auto" selects BF16 from Qwen3's config and fails at runtime.
        torch_dtype=torch.float16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    torch.cuda.synchronize(device_index)
    load_seconds = time.perf_counter() - load_start
    memory_after_load = torch.cuda.memory_allocated(device_index)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the knowledge-graph question using only the supplied "
                "evidence. Return only the answer entity, with no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evidence:\n(France, capital, Paris)\n\n"
                "Question: What is the capital of France? /no_think"
            ),
        },
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # The tokenizer bundled in the pinned AWQ revision predates Qwen's hard
    # switch and ignores enable_thinking.  Reproduce the current official
    # non-thinking template without modifying the downloaded checkpoint.
    manual_non_thinking_prefill = not prompt_text.endswith(
        NON_THINKING_ASSISTANT_PREFILL
    )
    if manual_non_thinking_prefill:
        prompt_text += NON_THINKING_ASSISTANT_PREFILL
    if not prompt_text.endswith(NON_THINKING_ASSISTANT_PREFILL):
        raise RuntimeError("Failed to construct the non-thinking prompt")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_tokens = int(inputs["input_ids"].shape[-1])

    torch.cuda.reset_peak_memory_stats(device_index)
    torch.cuda.synchronize(device_index)
    generation_start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize(device_index)
    generation_seconds = time.perf_counter() - generation_start

    output_ids = generated[0, input_tokens:]
    output_tokens = int(output_ids.shape[-1])
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    generation_peak = torch.cuda.max_memory_allocated(device_index)
    model_footprint = int(model.get_memory_footprint())

    normalized = " ".join(response.casefold().split())
    answer_check = "paris" in normalized
    thinking_disabled_check = "<think>" not in response and "</think>" not in response
    status = "PASS" if answer_check and thinking_disabled_check else "FAIL"

    payload: dict[str, object] = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_path": str(model_dir),
            "class": model.__class__.__name__,
            "dtype": str(model.dtype),
            "stored_parameter_tensor_elements": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "memory_footprint_mib": mib(model_footprint),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "autoawq": package_version("autoawq"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device_index),
        },
        "configuration": {
            "thinking": False,
            "soft_switch": "/no_think",
            "assistant_prefill": NON_THINKING_ASSISTANT_PREFILL,
            "manual_non_thinking_prefill": manual_non_thinking_prefill,
            "do_sample": False,
            "max_new_tokens": 16,
            "attention_implementation": "sdpa",
        },
        "timing_seconds": {
            "tokenizer_load": round(tokenizer_seconds, 6),
            "model_load": round(load_seconds, 6),
            "generation": round(generation_seconds, 6),
        },
        "memory_mib": {
            "before_load_allocated": mib(memory_before),
            "after_load_allocated": mib(memory_after_load),
            "generation_peak_allocated": mib(generation_peak),
        },
        "smoke_example": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response": response,
            "contains_expected_answer": answer_check,
            "thinking_disabled": thinking_disabled_check,
        },
    }
    write_json_atomic(output_path, payload)

    print(f"status={status}", flush=True)
    print(f"model_class={model.__class__.__name__}", flush=True)
    print(f"model_dtype={model.dtype}", flush=True)
    print(f"model_load_seconds={load_seconds:.3f}", flush=True)
    print(f"input_tokens={input_tokens} output_tokens={output_tokens}", flush=True)
    print(f"generation_seconds={generation_seconds:.3f}", flush=True)
    print(f"generation_peak_mib={mib(generation_peak):.3f}", flush=True)
    print(f"response={response!r}", flush=True)
    print(f"thinking_disabled={thinking_disabled_check}", flush=True)
    print(f"report={output_path}", flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"status=FAIL error={error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
