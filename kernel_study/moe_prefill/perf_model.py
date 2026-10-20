"""CPU only: issue / tensor / DRAM model of moe_prefill_tc per layer per rank (a prediction, not a measurement).

Per k-block instruction counts are the host SASS of the K = 3 bf16 instances (host_compile.sh: gu 2230 +
80 per active 16-row block + 1056 per warp with rows; dn 623 + 24 per block); tensor peak 1024 FLOP/clk/SM
(mma.sync m16n8k16 every 4 clk per SM), 48 SMs at 2.2 GHz, x1.15 for latency and barriers; DRAM 230 GB/s
with weights read once per active expert and the worst case for the output atomics. Each phase takes
max(compute, DRAM); + 0.8 ms glue (sort, zero-fill, work list).

  python3 kernel_study/moe_prefill/perf_model.py /home/sfxnz/projects/data/dsv41-1017/kernels/census_npy
"""
import array, random, struct, sys, math
CLK = 2.2e9; SMS = 48; EFF = 1.15; DRAM = 230e9
H, I, E, TOPK = 5120, 1152, 384, 6
KB1 = H // 128; KB2 = I // 128
def gu_unit_clk(rows, nb=2):
    mb = (rows + 15) // 16
    tot = 0
    # warp i on SMSP i % 4: warps (s, s+4)
    for s in range(4):
        t = 0
        for w in (s, s + 4):
            has = 8 * w < rows
            base = 2230 if nb == 2 else 1390
            t += base + (80 if nb == 2 else 48) * mb + (1056 if has else 0)
        tot = max(tot, t)
    tensor = 8 * (64 if nb == 2 else 32) * mb * 4
    kblk = max(tensor, tot) * EFF
    return KB1 * kblk + 8000 + 3000 * nb   # prologue + epilogue
def dn_unit_clk(rows):
    mb = (rows + 15) // 16
    issue = 2 * (623 + 24 * mb)
    tensor = 8 * 16 * mb * 4
    return KB2 * max(issue, tensor) * EFF + 7000
def chunks(n, mt):
    out = []
    while n > 0:
        out.append(min(mt, n)); n -= mt
    return out
def layer_ms(counts):
    g = sum(4 * gu_unit_clk(c, 2) + gu_unit_clk(c, 1) for n in counts for c in chunks(n, 64))
    d = sum(40 * dn_unit_clk(c) for n in counts for c in chunks(n, 128))
    t_gu = g / SMS / CLK * 1e3; t_dn = d / SMS / CLK * 1e3
    rows = sum(counts); active = sum(1 for n in counts if n)
    # DRAM: weights once per active expert (L2 reuse across an expert's chunks), x rows, A2 w+r, out atomics (worst: 2 x 4 B x rows x H)
    wbytes_gu = active * 2 * H * I * 3 / 8; wbytes_dn = active * I * H * 3 / 8
    by_gu = wbytes_gu + rows * H * 2 + rows * I * 2
    by_dn = wbytes_dn + rows * I * 2 + 2 * rows * H * 4
    return t_gu, t_dn, by_gu / DRAM * 1e3, by_dn / DRAM * 1e3
def load_census(path, layer):
    with open(path, 'rb') as f:
        f.read(6); f.read(2); hl = struct.unpack('<H', f.read(2))[0]; hdr = f.read(hl).decode()
        a = array.array('H'); a.frombytes(f.read())
    shape = eval(hdr.split("'shape':")[1].split(')')[0] + ')')
    n, L, k = shape
    return [[a[(i * L + layer) * k + j] for j in range(k)] for i in range(n)]
rng = random.Random(7)
census = []
import glob
for p in sorted(glob.glob(sys.argv[1] + '/*.npy')):
    census += load_census(p, 20)
print(f"census rows {len(census)}")
print("m      routing  rows/exp(max)  gu_ms  dn_ms  dram_gu dram_dn  model_ms   (glue +0.8 ms)")
for m in (512, 2048, 4096, 8192):
    for kind in ("uniform", "census"):
        cnt = [0] * E
        for t in range(m):
            ids = rng.sample(range(E), TOPK) if kind == "uniform" else census[rng.randrange(len(census))]
            for e in ids: cnt[e] += 1
        tg, td, dg, dd = layer_ms(cnt)
        tot = max(tg, dg) + max(td, dd) + 0.8
        print(f"{m:5d}  {kind:8s} {m*TOPK/E:6.0f}({max(cnt):5d})  {tg:6.1f} {td:6.1f}  {dg:6.1f}  {dd:6.1f}   {tot:6.1f}")
