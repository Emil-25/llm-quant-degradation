# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub"]
# ///
"""
Supervisor for the parallel eval fan-out. Run periodically. For each target model
it counts how many of its 12 cells (6 benchmarks x 2 precisions) are already in the
dataset, finds the latest HF Job for that model and its stage, and reports which
models are INCOMPLETE with no live job (i.e. need a resume-relaunch).

Jobs are resumable + idempotent (they skip cells already uploaded), so relaunching
a dead/incomplete model is always safe.

Output is plain lines; the last line is RELAUNCH:<comma-separated models or NONE>
so a wrapper can act on it.
"""
import os
import re
import sys
from huggingface_hub import HfApi, list_jobs

REPO = "Emil-7/llm-quant-degradation"
NS = "Emil-7"
BENCHMARKS = ["arc_challenge", "hellaswag", "gsm8k", "humaneval", "ifeval", "mgsm_direct"]
PRECISIONS = ["fp16", "int4_bnb"]
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it",
]
ACTIVE = {"RUNNING", "UPDATING", "SCHEDULED", "BUILDING"}

# Cells deliberately moved to the LOCAL GPU to save cloud budget (the slow int4
# HellaSwag MC runs for the big models). The supervisor must NOT flag these as
# needing a cloud relaunch, and the cloud target is 84 minus these.
DEFER_LOCAL = {
    ("qwen2.5-3b-instruct", "int4_bnb__hellaswag"),
    ("llama-3.2-3b-instruct", "int4_bnb__hellaswag"),
    ("gemma-2-2b-it", "int4_bnb__hellaswag"),
}


def short(m):
    return m.rstrip("/").split("/")[-1].lower()


def main():
    tok = os.environ["HF_TOKEN"]
    api = HfApi()
    files = {f for f in api.list_repo_files(REPO, repo_type="dataset", token=tok)
             if f.endswith(".json")}

    # latest job stage per model (parse --model from each job's command)
    latest = {}  # model_path -> (created_at, stage)
    for j in list_jobs(namespace=NS, token=tok):
        cmd = " ".join(j.command) if j.command else ""
        m = re.search(r"--model\s+(\S+)", cmd)
        if not m:
            continue
        mp = m.group(1)
        ca = getattr(j, "created_at", None)
        if mp not in latest or (ca and latest[mp][0] and ca > latest[mp][0]):
            latest[mp] = (ca, j.status.stage)

    cloud_target = 84 - len(DEFER_LOCAL)
    total_done = 0
    relaunch = []
    for mp in MODELS:
        s = short(mp)
        have = sum(1 for p in PRECISIONS for b in BENCHMARKS
                   if f"raw/{s}/{p}__{b}.json" in files)
        total_done += have
        # cells still missing that are NOT deferred to local
        missing_cloud = [f"{p}__{b}" for p in PRECISIONS for b in BENCHMARKS
                         if f"raw/{s}/{p}__{b}.json" not in files
                         and (s, f"{p}__{b}") not in DEFER_LOCAL]
        stage = latest.get(mp, (None, "NONE"))[1]
        live = stage in ACTIVE
        flag = ""
        if missing_cloud and not live:
            relaunch.append(mp)
            flag = "  <-- NEEDS RELAUNCH"
        elif not missing_cloud:
            flag = "  cloud-done"
        print(f"{s:24s} {have:2d}/12  job={stage:10s}{flag}")

    # progress against the cloud target (deferred-local cells excluded)
    cloud_done = total_done - sum(
        1 for (s, c) in DEFER_LOCAL if f"raw/{s}/{c}.json" not in files
    )
    print(f"TOTAL {total_done}/84 cells (cloud {cloud_done}/{cloud_target}; {len(DEFER_LOCAL)} deferred-local)")
    print("RELAUNCH:" + (",".join(relaunch) if relaunch else "NONE"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
