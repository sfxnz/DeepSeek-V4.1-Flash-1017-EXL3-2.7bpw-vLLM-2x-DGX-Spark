#!/usr/bin/env bash
# CPU only: compile the e15 p2b TU (build/chain_kbits.cu) and the e16 TU (build/chain_rows.cu, + widen_p2b_rows.py)
# for sm_121a in one no-GPU container; require every p2b kernel of the e15 TU (12 SORT=0/1, coop, dataflow and
# the 28 K-bits kernels) byte-identical in the e16 TU and the new kernels to be exactly the 9
# p2b_coop_dfw_kernel<BITS, 1>; print ptxas registers / spills / stack / smem of the dataflow kernels (e15 df
# next to the e16 wide copy) and the inner MMA-loop spill counts (kernel_study/p2b_coop/hot_loops.py).
#   kernel_study/p2b_rows/sass_identity_rows.sh [image]  > log      (~6-8 min: two nvcc -O3 of ~40-50 kernels)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${1:-dsv41-flash-exl3-sm121:e15-kbits}"
python3 "$HERE/make_bench_rows.py" >/dev/null
W="$HERE/build/identity"
rm -rf "$W" && mkdir -p "$W"
cp "$HERE/build/chain_kbits.cu" "$HERE/build/chain_rows.cu" "$W/"
set +e
docker run --rm --name dsv41-1017-rows-sass --network none --memory 12g --cpus 6 --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$W:/w" --entrypoint bash "$IMG" -c '
set -e
cd /w
EXL=/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext
T=/usr/local/lib/python3.12/dist-packages/torch/include
for f in chain_kbits chain_rows; do
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
[ "$rc" = 0 ] || { echo "sass_identity_rows: compile failed" >&2; exit 1; }
# SASS dump on the host (the image has no nvdisasm)
CUOBJDUMP="$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)"
mkdir -p "$W/sass"
for fn in $("$CUOBJDUMP" -symbols "$W/chain_rows.cubin" | grep -o "_Z[0-9]*p2b_coop_df[a-z]*_kernelILi[0-9]*ELi[0-9]*E[A-Za-z0-9_]*" | sort -u); do
  short=$(echo "$fn" | sed "s/EvPK6__half.*//")
  "$CUOBJDUMP" -sass -fun "$fn" "$W/chain_rows.cubin" > "$W/sass/$short.sass"
done
python3 - "$W" "$HERE/../p2b_srcsort" "$HERE/../p2b_coop" <<'PY'
import re, sys
from pathlib import Path
W = Path(sys.argv[1])
import importlib.util
def _load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem + "_rows", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
sections = _load(sys.argv[2] + "/text_identity.py").sections
inner_loops = _load(sys.argv[3] + "/hot_loops.py").inner_loops

KRE = re.compile(r"\.text\.(_Z\d+(p2b_moe_batched_kernel|p2b_coop_df_kernel|p2b_coop_dfw_kernel)ILi(\d+)ELi(\d+)(?:ELi(\d+))?E\w*)")
WIDE_BITS = (2, 3, 4, 23, 24, 32, 34, 42, 43)

def kernels(path):
    out = {}
    for sec, body in sections(str(path)).items():
        m = KRE.match(sec)
        if m:
            out[(m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5) or 0))] = (m.group(1), body)
    return out

def name(k):
    return f"{k[0]}<{k[1]}, {k[2]}" + (f", {k[3]}>" if k[0] == "p2b_moe_batched_kernel" else ">")

a, b = kernels(W / "chain_kbits.cubin"), kernels(W / "chain_rows.cubin")
bad = 0
print(f"== machine code of the e15 kernels in the e16 TU ({len(a)} kernels, expected 42)")
for k in sorted(a):
    same = k in b and a[k][1] == b[k][1]
    bad += not same
    print(f"{name(k)}: {len(a[k][1])} vs {len(b[k][1]) if k in b else 0} bytes {'IDENTICAL' if same else 'DIFFERENT'}")
want = {("p2b_coop_dfw_kernel", bits, 1, 0) for bits in WIDE_BITS}
new = set(b) - set(a)
print(f"== new kernels: {len(new)} (expected {len(want)})")
for k in sorted(new):
    print(f"  {name(k)}: {len(b[k][1])} bytes")
ok_new = new == want and len(a) == 42
if new != want:
    print(f"missing {sorted(want - new)} unexpected {sorted(new - want)}")

print("== ptxas resource usage (e16 TU), dataflow kernels: e15 df next to the e16 wide copy")
res = {}
cur = None
for l in (W / "chain_rows.nvcc.txt").read_text().splitlines():
    m = re.search(r"Compiling entry function '(\S+)'", l)
    if m:
        cur = m.group(1)
        continue
    if cur and "p2b_coop_df" in cur:
        m2 = KRE.match(".text." + cur)
        if not m2:
            continue
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
hot = {}
for k in sorted(res):
    sass = W / "sass" / (b[k][0].split("EvPK6__half")[0] + ".sass") if k in b else None
    loops = inner_loops(str(sass)) if sass and sass.exists() else []
    hot[k] = sum(lp["ldl"] + lp["stl"] for lp in loops), len(loops)
    hl = "; ".join(f"loop {lp['instr']} instr HMMA {lp['hmma']} LDL {lp['ldl']} STL {lp['stl']}" for lp in loops)
    r = res[k]
    print(f"{name(k)}: regs {r.get('regs')} stack {r.get('stack')} spill st/ld {r.get('spill_st')}/{r.get('spill_ld')} "
          f"smem {r.get('smem')} | {hl}")
# the wide copy keeps its launch bounds' registers (3 blocks/SM at K = 2, 2 otherwise) and spill-free MMA loops
for bits in WIDE_BITS:
    kd, kw = ("p2b_coop_df_kernel", bits, 1, 0), ("p2b_coop_dfw_kernel", bits, 1, 0)
    cap = 80 if bits == 2 else 128
    r = res.get(kw, {})
    if r.get("regs", 999) > cap or hot.get(kw, (99, 0))[0] > hot.get(kd, (0, 0))[0] or hot.get(kw, (0, 0))[1] != 2:
        print(f"WIDE <{bits}, 1>: regs {r.get('regs')} (cap {cap}), MMA-loop LDL+STL {hot.get(kw)} vs df {hot.get(kd)}: FAIL")
        bad += 1
ok = not bad and ok_new
print("sass_identity_rows:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
