// dsv41 moe_prefill_tc: torch binding and launchers for moe_prefill_tc.cuh (see that file and
// moe_prefill_tc.py). Built ahead of time by docker/Dockerfile.e16p, or at first use with
// torch.utils.cpp_extension (moe_prefill_tc.py), against ExLlamaV3's extension headers.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <mutex>
#include <unordered_map>

#include "moe_prefill_tc.cuh"

namespace {

using namespace dsv41_tc;

#define TC_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be a contiguous CUDA tensor")

void check_cuda(cudaError_t err, const char* what)
{
    TORCH_CHECK(err == cudaSuccess, "moe_prefill_tc: ", what, ": ", cudaGetErrorString(err));
}

// Persistent grid size per kernel instance: resident CTAs per SM x SMs (opt-in smem set once).
template <typename Kernel>
int persistent_grid(Kernel kernel, int smem, int device)
{
    static std::mutex mtx;
    static std::unordered_map<const void*, int> cache[64];
    std::lock_guard<std::mutex> lock(mtx);
    auto& c = cache[device & 63];
    auto it = c.find((const void*) kernel);
    if (it != c.end()) return it->second;
    check_cuda(cudaFuncSetAttribute((const void*) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
               "cudaFuncSetAttribute");
    int per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, smem), "occupancy");
    TORCH_CHECK(per_sm >= 1, "moe_prefill_tc: kernel does not fit on an SM (smem ", smem, " B)");
    int sms = 0;
    check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "SM count");
    const int grid = per_sm * sms;
    c.emplace((const void*) kernel, grid);
    return grid;
}

template <int KG, bool BF16>
void launch_gu(int device, cudaStream_t stream, const void* x, int hidden, int inter, const Tables& tb,
               const int* tok, const int4* items, int* ctrl, half* a2, float limit)
{
    auto kernel = gu_kernel<KG, BF16>;
    const int smem = gu_smem_bytes(KG);
    const int grid = persistent_grid(kernel, smem, device);
    kernel<<<grid, kThreads, smem, stream>>>(x, hidden, inter, tb, tok, items, ctrl, a2, limit);
}

template <int KD>
void launch_dn(int device, cudaStream_t stream, int hidden, int inter, const Tables& tb, const int* tok,
               const half* w, const int4* items, int* ctrl, const half* a2, float* out)
{
    auto kernel = dn_kernel<KD>;
    const int smem = dn_smem_bytes(KD);
    const int grid = persistent_grid(kernel, smem, device);
    kernel<<<grid, kThreads, smem, stream>>>(hidden, inter, tb, tok, w, items, ctrl, a2, out);
}

template <typename T>
const T* const* ptr_table(const at::Tensor& t)
{
    return reinterpret_cast<const T* const*>(t.data_ptr());
}

}  // namespace

// out[T, hidden] (fp32, zeroed by the caller) += routed experts of x. Rows are pre-sorted by local expert
// (stable): token_sorted / weight_sorted [T * topk], counts [n_exp + 1] (last bin: invalid ids, skipped).
// a2 [T * topk, inter] fp16 scratch; items1 / items2 int32 [cap, 4]; ctrl int32 [4]. Pointer tables are
// vllm_exl3's layer._exl3_ptrs (int64 [n_exp]). MCG codebook only; K_gu and K_d in 2..4.
void moe_prefill_tc(at::Tensor x, at::Tensor out, at::Tensor token_sorted, at::Tensor weight_sorted,
                    at::Tensor counts, at::Tensor gate_t, at::Tensor gate_suh, at::Tensor gate_svh,
                    at::Tensor up_t, at::Tensor up_suh, at::Tensor up_svh, at::Tensor down_t,
                    at::Tensor down_suh, at::Tensor down_svh, at::Tensor a2, at::Tensor items1,
                    at::Tensor items2, at::Tensor ctrl, int64_t k_gu, int64_t k_d, double limit)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    for (const at::Tensor* t : {&x, &out, &token_sorted, &weight_sorted, &counts, &gate_t, &gate_suh, &gate_svh,
                                &up_t, &up_suh, &up_svh, &down_t, &down_suh, &down_svh, &a2, &items1, &items2, &ctrl})
        TC_CHECK_CUDA(*t);
    TORCH_CHECK(x.dim() == 2 && (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16),
                "x must be [T, hidden] fp16 or bf16");
    const int64_t tokens = x.size(0);
    const int hidden = (int) x.size(1);
    const int inter = (int) a2.size(1);
    const int n_exp = (int) counts.size(0) - 1;
    TORCH_CHECK(hidden % 128 == 0 && inter % 128 == 0 && hidden >= 128 && inter >= 128, "hidden and inter must be multiples of 128");
    TORCH_CHECK(n_exp >= 1 && n_exp <= 1024, "1..1024 local experts");
    TORCH_CHECK(out.scalar_type() == at::kFloat && out.dim() == 2 && out.size(0) == tokens && out.size(1) == hidden, "out must be [T, hidden] fp32");
    TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
    const int64_t slots = token_sorted.numel();
    TORCH_CHECK(token_sorted.scalar_type() == at::kInt && weight_sorted.scalar_type() == at::kHalf && weight_sorted.numel() == slots,
                "token_sorted int32 / weight_sorted fp16, same length");
    TORCH_CHECK(slots < (int64_t) 1 << 31 && tokens * hidden < ((int64_t) 1 << 40), "too many rows");
    TORCH_CHECK(a2.scalar_type() == at::kHalf && a2.dim() == 2 && a2.size(0) >= slots, "a2 must be [>= T * topk, inter] fp16");
    TORCH_CHECK(items1.scalar_type() == at::kInt && items2.scalar_type() == at::kInt && ctrl.scalar_type() == at::kInt, "int32 work list");
    TORCH_CHECK(items1.numel() >= 4 * ((slots + kMt1 - 1) / kMt1 + n_exp), "items1 too small");
    TORCH_CHECK(items2.numel() >= 4 * ((slots + kMt2 - 1) / kMt2 + n_exp), "items2 too small");
    TORCH_CHECK(ctrl.numel() >= kCtrlInts, "ctrl too small");
    for (const at::Tensor* t : {&gate_t, &gate_suh, &gate_svh, &up_t, &up_suh, &up_svh, &down_t, &down_suh, &down_svh})
        TORCH_CHECK(t->scalar_type() == at::kLong && t->numel() == n_exp, "pointer tables must be int64 [n_exp]");
    TORCH_CHECK(k_gu >= 2 && k_gu <= 4 && k_d >= 2 && k_d <= 4, "K_gu and K_d must be 2..4");

    Tables tb;
    tb.gate_trellis = ptr_table<uint16_t>(gate_t);
    tb.gate_suh = ptr_table<half>(gate_suh);
    tb.gate_svh = ptr_table<half>(gate_svh);
    tb.up_trellis = ptr_table<uint16_t>(up_t);
    tb.up_suh = ptr_table<half>(up_suh);
    tb.up_svh = ptr_table<half>(up_svh);
    tb.down_trellis = ptr_table<uint16_t>(down_t);
    tb.down_suh = ptr_table<half>(down_suh);
    tb.down_svh = ptr_table<half>(down_svh);

    const int device = x.get_device();
    int4* it1 = reinterpret_cast<int4*>(items1.data_ptr());
    int4* it2 = reinterpret_cast<int4*>(items2.data_ptr());
    int* cptr = ctrl.data_ptr<int>();
    const int* tok = token_sorted.data_ptr<int>();
    half* a2p = reinterpret_cast<half*>(a2.data_ptr());
    const float lim = limit > 0.0 ? (float) limit : 0.0f;

    worklist_kernel<<<1, 1024, 0, stream>>>(counts.data_ptr<int64_t>(), n_exp, it1, it2, cptr);
    check_cuda(cudaPeekAtLastError(), "worklist launch");

    const bool bf16 = x.scalar_type() == at::kBFloat16;
    const void* xp = x.data_ptr();
#define GU(K)                                                                                          \
    (bf16 ? launch_gu<K, true>(device, stream, xp, hidden, inter, tb, tok, it1, cptr, a2p, lim)       \
          : launch_gu<K, false>(device, stream, xp, hidden, inter, tb, tok, it1, cptr, a2p, lim))
    switch (k_gu) {
        case 2: GU(2); break;
        case 3: GU(3); break;
        default: GU(4); break;
    }
#undef GU
    check_cuda(cudaPeekAtLastError(), "gate/up launch");

    const half* wp = reinterpret_cast<const half*>(weight_sorted.data_ptr());
    float* op = out.data_ptr<float>();
    switch (k_d) {
        case 2: launch_dn<2>(device, stream, hidden, inter, tb, tok, wp, it2, cptr, a2p, op); break;
        case 3: launch_dn<3>(device, stream, hidden, inter, tb, tok, wp, it2, cptr, a2p, op); break;
        default: launch_dn<4>(device, stream, hidden, inter, tb, tok, wp, it2, cptr, a2p, op); break;
    }
    check_cuda(cudaPeekAtLastError(), "down launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_prefill_tc", &moe_prefill_tc,
          "routed-expert MoE prefill, EXL3 MCG decoded once per (expert, row chunk), mma.sync fp32 accumulate");
    m.attr("ABI") = 1;
    m.attr("MT1") = kMt1;
    m.attr("MT2") = kMt2;
}
