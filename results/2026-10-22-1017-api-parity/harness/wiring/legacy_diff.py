"""API_GATEWAY=0 must reproduce the base commit's head and worker `docker run` exactly.

Runs tests/run_sh_harness.py's dry run twice with the same ports: once on the base run.sh (01e170c, copied
to legacy/run.sh) and once on the worktree's run.sh with API_GATEWAY=0; temp paths are normalized.
Then prints how the API_GATEWAY=1 commands differ from the legacy ones.
"""
import difflib
import shlex
import sys
from pathlib import Path

WT = Path("/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-api-wiring")
LEGACY = Path(__file__).resolve().parent / "legacy"
sys.path.insert(0, str(WT / "tests"))
import run_sh_harness as h  # noqa: E402

PORTS = {"PORT": "18000", "VLLM_PORT": "18001"}


def run(root: Path, **extra):
    h.ROOT = root
    res = h.dry_run(**PORTS, **extra)
    assert res["returncode"] == 0, res["stdout"] + res["stderr"]
    tmp = str(Path(res["env"]["HOME"]))
    norm = lambda argv: [a.replace(tmp, "<tmp>") for a in argv] if argv else argv
    ssh = [c["argv"][-1].replace(tmp, "<tmp>") for c in res["calls"]
           if c["tool"] == "ssh" and c["argv"][-1].endswith("bash /tmp/dsv41-exl3-run.sh") and "PREFLIGHT" not in c["argv"][-1]]
    return {"head": norm(res["head"]), "worker": norm(res["worker"]), "gateway": norm(res["gateway"]), "ssh": ssh}


base = run(LEGACY)
off = run(WT, API_GATEWAY="0")
on = run(WT)
for role in ("head", "worker"):
    same = base[role] == off[role]
    print(f"API_GATEWAY=0 {role} docker run == base {role}: {same} ({len(off[role])} argv items)")
print(f"API_GATEWAY=0 starts a gateway container: {off['gateway'] is not None}")
print("API_GATEWAY=0 worker ssh line vs base (not a docker command):")
for line in difflib.unified_diff(shlex.split(base["ssh"][0]), shlex.split(off["ssh"][0]), lineterm="", n=0):
    if line[:1] in "+-" and not line.startswith(("+++", "---")):
        print("   ", line)
for role in ("head", "worker"):
    print(f"API_GATEWAY=1 {role} vs base, argv items:")
    for line in difflib.unified_diff(base[role], on[role], lineterm="", n=1):
        if not line.startswith(("+++", "---")):
            print("   ", line)
print("API_GATEWAY=1 gateway:\n    docker " + shlex.join(on["gateway"]))
