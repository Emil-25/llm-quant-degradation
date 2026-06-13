"""Diagnostic: does SmolLM2 fail HumanEval because it can't code, or because it
derails on RAW completion? Compares raw vs chat-templated generation on one
hardcoded HumanEval prompt. Generation only (no code exec) -> runs on Windows.
Avoids the `datasets` loader (which segfaults here) by hardcoding the prompt."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda")

# RAW completion (the protocol the eval used)
ids = tok(PROMPT, return_tensors="pt").to("cuda")
out = model.generate(**ids, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
raw = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
print("=== RAW COMPLETION (what eval used) ===")
print(repr(raw[:500]))

# CHAT-TEMPLATE variant
msg = [{"role": "user", "content": "Complete this Python function (return code only):\n\n" + PROMPT}]
cids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to("cuda")
cout = model.generate(cids, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
chat = tok.decode(cout[0][cids.shape[1]:], skip_special_tokens=True)
print("\n=== CHAT-TEMPLATE COMPLETION ===")
print(repr(chat[:500]))
