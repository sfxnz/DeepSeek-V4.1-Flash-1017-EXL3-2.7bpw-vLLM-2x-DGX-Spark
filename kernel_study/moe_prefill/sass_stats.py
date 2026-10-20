#!/usr/bin/env python3
"""ptxas registers / spills per moe_prefill_tc kernel and the instruction mix of each kernel's hot loop
(the largest backward-branch region with HMMA). Exit 1 on any spill or collective-emulation (WARPSYNC)
in the gate/up and down kernels. Used by host_compile.sh."""

from __future__ import annotations

import collections
import re
import sys

KEYS = ("HMMA", "LDSM", "SHFL", "FFMA", "FADD", "LOP3", "IMAD", "PRMT", "HADD2", "SHF", "LDS", "STS", "LDG", "LDGSTS", "BAR")


def kernel_label(name: str) -> str:
    m = re.search(r"(gu_kernel|dn_kernel|worklist_kernel)(ILi(\d)E(Lb([01]))?)?", name)
    if not m:
        return name[:40]
    label = m.group(1)
    if m.group(3):
        label += f" K={m.group(3)}"
    if m.group(5) is not None:
        label += " bf16" if m.group(5) == "1" else " fp16"
    return label


def main(ptxas_log: str, sass: str) -> int:
    bad = 0
    name = None
    for line in open(ptxas_log):
        m = re.search(r"entry function '(\S+)'", line)
        if m:
            name = m.group(1)
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and name and (int(m.group(1)) or int(m.group(2))):
            print(f"SPILL {kernel_label(name)}: {line.strip()}")
            bad = 1
        m = re.search(r"Used (\d+) registers", line)
        if m and name:
            print(f"{kernel_label(name):24s} {m.group(1)} registers")
    for f in re.split(r"\n\s*Function : ", open(sass).read())[1:]:
        fname = f.split("\n", 1)[0].strip()
        ins = []
        for l in f.split("\n"):
            m = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)(.*)", l)
            if m:
                ins.append((int(m.group(1), 16), m.group(3).split(".")[0], m.group(4)))
        ops = collections.Counter(o for _, o, _ in ins)
        label = kernel_label(fname)
        if ops["WARPSYNC"] and not label.startswith("worklist"):
            print(f"COLLECTIVE EMULATION {label}: {ops['WARPSYNC']} WARPSYNC")
            bad = 1
        addr = {a: i for i, (a, _, _) in enumerate(ins)}
        loops = []
        for i, (_, o, r) in enumerate(ins):
            if o != "BRA":
                continue
            m = re.search(r"0x([0-9a-f]+)", r)
            if m and int(m.group(1), 16) in addr and addr[int(m.group(1), 16)] < i:
                seg = ins[addr[int(m.group(1), 16)]: i + 1]
                c = collections.Counter(o for _, o, _ in seg)
                if c["HMMA"]:
                    loops.append((len(seg), c))
        # Innermost first: the K loops (one k-block per iteration; gu has one per column variant).
        for n, c in sorted(loops, key=lambda t: t[0])[:2]:
            print(f"{label:24s} loop {n} instr, {c['HMMA']} HMMA ({n / c['HMMA']:.1f}/HMMA): "
                  + " ".join(f"{k} {c[k]}" for k in KEYS if c[k]))
    return bad


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
