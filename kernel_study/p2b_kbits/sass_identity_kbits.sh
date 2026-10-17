#!/usr/bin/env bash
# CPU only: compile the canonical-e14 p2b TU (build/chain_r3.cu) and the e15 TU (build/chain_kbits.cu,
# + widen_p2b_kbits.py) for sm_121a in one no-GPU container; require every kernel of the e14 TU
# (12 p2b SORT=0/1, <2, 1, 2> coop, p2b_coop_df_kernel<2, 1>) byte-identical in the e15 TU and the new
# kernels to be exactly the expected K-bits set; print ptxas registers / spills / stack of every p2b
# kernel and the inner MMA-loop spill counts (hot_loops.py) of the coop and dataflow kernels.
#   kernel_study/p2b_kbits/sass_identity_kbits.sh [image]  > log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${1:-dsv41-flash-exl3-sm121:canonical-e14}"
python3 "$HERE/make_bench_kbits.py" >/dev/null
W="$HERE/build/identity"
rm -rf "$W" && mkdir -p "$W"
cp "$HERE/build/chain_r3.cu" "$HERE/build/chain_kbits.cu" "$W/"
set +e
docker run --rm --name dsv41-1017-kb-sass --network none --memory 12g --cpus 6 --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$W:/w" --entrypoint bash "$IMG" -c '
set -e
cd /w
EXL=/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext
T=/usr/local/lib/python3.12/dist-packages/torch/include
for f in chain_r3 chain_kbits; do
  t0=$(date +%s)
  nvcc -cubin -o $f.cubin $f.cu -std=c++17 -O3 -gencode arch=compute_121a,code=sm_121a \
    -I$EXL -I$T -I$T/torch/csrc/api/include -I/usr/include/python3.12 \
    -DTORCH_EXTENSION_NAME=p2b_ident -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    --expt-relaxed-constexpr -Xptxas -v 2> $f.nvcc.txt || { grep -B2 -A2 " error" $f.nvcc.txt | head -40; exit 1; }
  echo "$f: nvcc $(( $(date +%s) - t0 )) s"
done
nvcc --version | tail -1
' 2>&1 | grep -v 'vllm._C\|cpp_extension.py\|^$'
rc=${PIPESTATUS[0]}
set -e
[ "$rc" = 0 ] || { echo "sass_identity_kbits: compile failed" >&2; exit 1; }
# SASS dump on the host (the image has no nvdisasm)
CUOBJDUMP="$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)"
mkdir -p "$W/sass"
for fn in $("$CUOBJDUMP" -symbols "$W/chain_kbits.cubin" | grep -o "_Z[0-9]*p2b_[a-z0-9_]*kernelILi[0-9]*ELi[0-9]*E[A-Za-z0-9_]*" | sort -u); do
  short=$(echo "$fn" | sed "s/EvPK6__half.*//")
  "$CUOBJDUMP" -sass -fun "$fn" "$W/chain_kbits.cubin" > "$W/sass/$short.sass"
done
python3 - "$W" "$HERE/../p2b_srcsort" "$HERE/../p2b_coop" <<'PY'
import re, sys
from pathlib import Path
W = Path(sys.argv[1])
import importlib.util
def _load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem + "_kb", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
sections = _load(sys.argv[2] + "/text_identity.py").sections
inner_loops = _load(sys.argv[3] + "/hot_loops.py").inner_loops

KRE = re.compile(r"\.text\.(_Z\d+(p2b_moe_batched_kernel|p2b_coop_df_kernel)ILi(\d+)ELi(\d+)(?:ELi(\d+))?E\w*)")

def kernels(path):
    out = {}
    for sec, body in sections(str(path)).items():
        m = KRE.match(sec)
        if m:
            out[(m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5) or 0))] = (m.group(1), body)
    return out

def name(k):
    return f"{k[0]}<{k[1]}, {k[2]}" + (f", {k[3]}>" if k[0] == "p2b_moe_batched_kernel" else ">")

a, b = kernels(W / "chain_r3.cubin"), kernels(W / "chain_kbits.cubin")
bad = 0
print("== machine code of the canonical-e14 kernels in the e15 TU")
for k in sorted(a):
    same = k in b and a[k][1] == b[k][1]
    bad += not same
    print(f"{name(k)}: {len(a[k][1])} vs {len(b[k][1]) if k in b else 0} bytes {'IDENTICAL' if same else 'DIFFERENT'}")
mixed = (23, 24, 32, 34, 42, 43)
want = {("p2b_moe_batched_kernel", bits, 1, s) for bits in mixed for s in (0, 1, 2)}
want |= {("p2b_moe_batched_kernel", bits, 1, 2) for bits in (3, 4)}
want |= {("p2b_coop_df_kernel", bits, 1, 0) for bits in (3, 4) + mixed}
new = set(b) - set(a)
print(f"== new kernels: {len(new)} (expected {len(want)})")
for k in sorted(new):
    print(f"  {name(k)}: {len(b[k][1])} bytes")
ok_new = new == want
if not ok_new:
    print(f"missing {sorted(want - new)} unexpected {sorted(new - want)}")

print("== ptxas resource usage (e15 TU), every p2b kernel")
lines = (W / "chain_kbits.nvcc.txt").read_text().splitlines()
cur = None
res = {}
for l in lines:
    m = re.search(r"Compiling entry function '(\S+)'", l)
    if m:
        cur = m.group(1)
        continue
    if cur and ("p2b_moe_batched_kernel" in cur or "p2b_coop_df_kernel" in cur):
        m2 = KRE.match(".text." + cur)
        key = (m2.group(2), int(m2.group(3)), int(m2.group(4)), int(m2.group(5) or 0))
        r = res.setdefault(key, {})
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", l)
        if m:
            r.update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
        m = re.search(r"Used (\d+) registers", l)
        if m:
            r["regs"] = int(m.group(1))
            m3 = re.search(r"(\d+) bytes smem", l)
            r["smem"] = int(m3.group(1)) if m3 else None
for k in sorted(res):
    sass = W / "sass" / (b[k][0].split("EvPK6__half")[0] + ".sass") if k in b else None
    loops = inner_loops(str(sass)) if sass and sass.exists() else []
    hl = "; ".join(f"loop {lp['instr']} instr HMMA {lp['hmma']} LDL {lp['ldl']} STL {lp['stl']}" for lp in loops)
    r = res[k]
    print(f"{name(k)}: regs {r.get('regs')} stack {r.get('stack')} spill st/ld {r.get('spill_st')}/{r.get('spill_ld')} "
          f"smem {r.get('smem')} | {hl}")
ok = not bad and ok_new
print("sass_identity_kbits:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
