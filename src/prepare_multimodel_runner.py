#!/usr/bin/env python3
"""Create a cross-model runner from the frozen Qwen3 evaluation script.

The transformation is deliberately narrow and fails closed if the frozen
source differs from the archived, hash-verified version.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "43e4b4f37e874836b21fec13c6778649db3dbbaaabde3705db4bfeeea128b603"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Frozen source hash mismatch: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )

    text = source_bytes.decode("utf-8")
    text = replace_once(
        text,
        '"""Reproducible Qwen3-AWQ evaluation for budgeted MetaQA path evidence."""',
        '"""Reproducible cross-model evaluation for budgeted MetaQA path evidence."""',
        "module docstring",
    )
    text = replace_once(
        text,
        (
            'MODEL_ID = "Qwen/Qwen3-8B-AWQ"\n'
            'MODEL_REVISION = "cb7d6a337aadb4d2082ed0dcef1032e4f8645194"\n'
        ),
        "",
        "remove Qwen-only metadata constants",
    )
    text = replace_once(
        text,
        "def build_prompt(tokenizer, question: str, evidence: str) -> tuple[str, bool]:",
        (
            "def build_prompt(\n"
            "    tokenizer, question: str, evidence: str, prompt_family: str\n"
            ") -> tuple[str, bool]:"
        ),
        "build_prompt signature",
    )
    text = replace_once(
        text,
        '                f"Question: {question}\\n/no_think"\n',
        (
            '                f"Question: {question}"\n'
            '                + ("\\n/no_think" if prompt_family == "qwen3" else "")\n'
        ),
        "Qwen-only no-think suffix",
    )
    old_template = '''    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    manual_prefill = not prompt.endswith(NON_THINKING_ASSISTANT_PREFILL)
    if manual_prefill:
        prompt += NON_THINKING_ASSISTANT_PREFILL
    if not prompt.endswith(NON_THINKING_ASSISTANT_PREFILL):
        raise RuntimeError("Failed to construct the non-thinking prompt")
    return prompt, manual_prefill
'''
    new_template = '''    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if prompt_family == "qwen3":
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    manual_prefill = False
    if prompt_family == "qwen3":
        manual_prefill = not prompt.endswith(NON_THINKING_ASSISTANT_PREFILL)
        if manual_prefill:
            prompt += NON_THINKING_ASSISTANT_PREFILL
        if not prompt.endswith(NON_THINKING_ASSISTANT_PREFILL):
            raise RuntimeError("Failed to construct the non-thinking prompt")
    return prompt, manual_prefill
'''
    text = replace_once(text, old_template, new_template, "chat template branch")

    text = replace_once(
        text,
        '    parser.add_argument("--model-dir", type=Path, required=True)\n',
        (
            '    parser.add_argument("--model-dir", type=Path, required=True)\n'
            '    parser.add_argument("--model-id", required=True)\n'
            '    parser.add_argument("--model-revision", required=True)\n'
            '    parser.add_argument(\n'
            '        "--model-dtype", choices=("float16", "bfloat16"), required=True\n'
            '    )\n'
            '    parser.add_argument(\n'
            '        "--prompt-family", choices=("qwen3", "generic"), required=True\n'
            '    )\n'
            '    parser.add_argument("--trust-remote-code", action="store_true")\n'
        ),
        "model CLI arguments",
    )
    text = replace_once(
        text,
        '''    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
''',
        '''    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
''',
        "tokenizer trust setting",
    )
    text = replace_once(
        text,
        '''    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
''',
        '''    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
''',
        "model trust setting",
    )
    text = replace_once(
        text,
        "    print(\"loading_model...\", flush=True)\n"
        "    load_started = time.perf_counter()\n",
        (
            "    print(\"loading_model...\", flush=True)\n"
            "    model_dtype = {\n"
            "        \"float16\": torch.float16,\n"
            "        \"bfloat16\": torch.bfloat16,\n"
            "    }[args.model_dtype]\n"
            "    load_started = time.perf_counter()\n"
        ),
        "dtype mapping",
    )
    text = replace_once(
        text,
        "        torch_dtype=torch.float16,\n",
        "        torch_dtype=model_dtype,\n",
        "model dtype",
    )
    text = replace_once(
        text,
        '''    warmup_prompt, _ = build_prompt(
        tokenizer,
        "What is the capital of France?",
        "- France --[capital]--> Paris",
    )
''',
        '''    warmup_prompt, _ = build_prompt(
        tokenizer,
        "What is the capital of France?",
        "- France --[capital]--> Paris",
        args.prompt_family,
    )
''',
        "warmup prompt",
    )
    text = replace_once(text, '            "id": MODEL_ID,\n', '            "id": args.model_id,\n', "model id")
    text = replace_once(
        text,
        '            "revision": MODEL_REVISION,\n',
        '            "revision": args.model_revision,\n',
        "model revision",
    )
    text = replace_once(
        text,
        '            "thinking": False,\n',
        (
            '            "thinking": False,\n'
            '            "prompt_family": args.prompt_family,\n'
            '            "model_dtype_requested": args.model_dtype,\n'
            '            "trust_remote_code": args.trust_remote_code,\n'
        ),
        "configuration metadata",
    )
    text = replace_once(
        text,
        '            "prompt_protocol": "endpoint_instruction_v3_frozen_retrieval",\n',
        (
            '            "prompt_protocol": (\n'
            '                "endpoint_instruction_v3_frozen_retrieval"\n'
            '                if args.prompt_family == "qwen3"\n'
            '                else "endpoint_instruction_v3_cross_model_generic"\n'
            '            ),\n'
        ),
        "prompt protocol metadata",
    )
    text = replace_once(
        text,
        "            prompt, manual_prefill = build_prompt(tokenizer, question, evidence)\n",
        (
            "            prompt, manual_prefill = build_prompt(\n"
            "                tokenizer, question, evidence, args.prompt_family\n"
            "            )\n"
        ),
        "evaluation prompt",
    )

    compile(text, str(args.output), "exec")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(args.output)
    output_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"source_sha256={source_sha256}")
    print(f"output={args.output.resolve()}")
    print(f"output_sha256={output_sha256}")
    print("status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
