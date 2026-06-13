# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.6.0",
#   "transformers==4.49.0",
#   "accelerate",
#   "bitsandbytes",
#   "human-eval",
#   "numpy",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
# ///
"""
Chat-template HumanEval with proper code extraction + execution, for instruct
models that derail on raw completion (e.g. SmolLM2). Run under WSL/Linux (the
human-eval execution sandbox is Linux-only).

Standard lm-eval HumanEval concatenates prompt+completion, which is wrong for a
chat model that emits a FULL function. Here we: prompt via chat template, extract
the model's complete function (from a ```python block if present), then execute it
against the problem's tests via human-eval's check_correctness with an empty
prompt (so the program is exactly the model's function + tests).

Usage (in WSL):
    uv run he_chat_eval.py --precision fp16   --limit 0
    uv run he_chat_eval.py --precision int4_bnb --limit 0
"""
import argparse
import re
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import read_problems
from human_eval.execution import check_correctness


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        # prefer the longest code block
        return max(blocks, key=len)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "int4_bnb"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all 164 problems")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = dict(torch_dtype=torch.float16, device_map="cuda")
    if args.precision == "int4_bnb":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()

    problems = read_problems()
    items = list(problems.items())
    if args.limit:
        items = items[: args.limit]

    import sys
    passed = 0
    for i, (tid, prob) in enumerate(items):
        try:
            msg = [{"role": "user", "content":
                    "Complete this Python function. Respond with ONLY the complete "
                    "function inside a ```python code block:\n\n" + prob["prompt"]}]
            ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=512, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            code = extract_code(text)
            fake = {"task_id": tid, "prompt": "", "test": prob["test"],
                    "entry_point": prob["entry_point"]}
            res = check_correctness(fake, code, timeout=5.0)
            passed += int(res["passed"])
        except Exception as e:
            print(f"WARN problem {tid} failed: {type(e).__name__}: {e}", file=sys.stderr)
        if (i + 1) % 40 == 0:
            print(f"...{i+1}/{len(items)} done, passed={passed}", file=sys.stderr, flush=True)

    n = len(items)
    pass1 = round(100.0 * passed / n, 2)
    result = {"model": args.model, "precision": args.precision,
              "n": n, "passed": passed, "pass@1": pass1}
    out_path = f"results/smollm2_he_{args.precision}.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
