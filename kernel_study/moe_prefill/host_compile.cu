// Host-only compile of docker/patch/moe_prefill_tc.cuh (no torch): every kernel instance the binding
// launches, for ptxas resource checks (registers, spills, smem). kernel_study/moe_prefill/host_compile.sh.
#include "moe_prefill_tc.cuh"

using namespace dsv41_tc;

template __global__ void dsv41_tc::gu_kernel<2, true>(const void*, int, int, Tables, const int*, const int4*, int*, half*, float);
template __global__ void dsv41_tc::gu_kernel<3, true>(const void*, int, int, Tables, const int*, const int4*, int*, half*, float);
template __global__ void dsv41_tc::gu_kernel<4, true>(const void*, int, int, Tables, const int*, const int4*, int*, half*, float);
template __global__ void dsv41_tc::gu_kernel<2, false>(const void*, int, int, Tables, const int*, const int4*, int*, half*, float);
template __global__ void dsv41_tc::gu_kernel<3, false>(const void*, int, int, Tables, const int*, const int4*, int*, half*, float);
template __global__ void dsv41_tc::gu_kernel<4, false>(const void*, int, int, Tables, const int*, const int4*, int*, half*, float);
template __global__ void dsv41_tc::dn_kernel<2>(int, int, Tables, const int*, const half*, const int4*, int*, const half*, float*);
template __global__ void dsv41_tc::dn_kernel<3>(int, int, Tables, const int*, const half*, const int4*, int*, const half*, float*);
template __global__ void dsv41_tc::dn_kernel<4>(int, int, Tables, const int*, const half*, const int4*, int*, const half*, float*);
