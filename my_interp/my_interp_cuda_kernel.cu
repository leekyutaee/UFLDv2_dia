/********************************************************************
 * my_interp_cuda_kernel.cu  ―  block = (1 batch, 1 lane)
 *******************************************************************/
#include <torch/extension.h>
using namespace at;

template <typename scalar_t>
__global__ void interp_radius20_kernel(
        PackedTensorAccessor<scalar_t,4,RestrictPtrTraits,size_t>  in,     // [bs, lanes, 125, 2]
        PackedTensorAccessor<scalar_t,2,RestrictPtrTraits,size_t>  coords, // [N, 2]
        PackedTensorAccessor<scalar_t,3,RestrictPtrTraits,size_t>  out,    // [bs, N, 2]
        const float radius_sq)                                            // (20px)^2
{
    const int bs    = in.size(0);
    const int lanes = in.size(1);
    const int N     = coords.size(0);

    /* ── block ↔ (batch, lane) 매핑 ───────────────────────────*/
    const int bIdx = blockIdx.x;          // 0 … bs-1
    const int lIdx = blockIdx.y;          // 0 … lanes-1
    if (bIdx >= bs || lIdx >= lanes) return;

    /* ── 이 block 의 thread 들이 125 point 를 stride 방식으로 분담 ─*/
    for (int pIdx = threadIdx.x; pIdx < 125; pIdx += blockDim.x)
    {
        scalar_t x1 = in[bIdx][lIdx][pIdx][0];
        scalar_t y1 = in[bIdx][lIdx][pIdx][1];
        if (x1 < 0 || y1 < 0) continue;          // 무효 포인트 skip

        /* ─ coords[] 2700 개 전부 검사 ───────────────────────*/
        for (int j = 0; j < N; ++j) {
            scalar_t xq = coords[j][0];
            scalar_t yq = coords[j][1];
            if (xq < 0 || yq < 0) continue;

            scalar_t dx = xq - x1,  dy = yq - y1;
            if (dx*dx + dy*dy <= radius_sq) {
                /* 최초 기록만 허용 (atomic 필요 X: 반드시 음수→양수 한번) */
                if (out[bIdx][j][0] < 0) {
                    out[bIdx][j][0] = xq;
                    out[bIdx][j][1] = yq;
                }
                break;          // 같은 포인트에서 coords 더 볼 필요 없음
            }
        }
    }
}

/* ────────── 런처 ────────── */
Tensor interp_radius20_launcher(const Tensor& input,
                                const Tensor& coords)
{
    TORCH_CHECK(input.is_cuda() && coords.is_cuda(), "CUDA tensors expected");
    const int bs    = input.size(0);
    const int lanes = input.size(1);
    const int N     = coords.size(0);

    auto out = at::full({bs, N, 2}, -1.0f,
                        input.options().dtype(kFloat).device(kCUDA));

    /* grid  = (bs , lanes) ,  block = 32 threads  */
    dim3 grid(bs, lanes);
    dim3 block(32);                    // 32·warp 단위, 32|64 둘 다 OK

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "interp_radius20_kernel", ([&]{
        interp_radius20_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor<scalar_t,4,RestrictPtrTraits,size_t>(),
            coords.packed_accessor<scalar_t,2,RestrictPtrTraits,size_t>(),
            out.packed_accessor<scalar_t,3,RestrictPtrTraits,size_t>(),
            100.0f);   // 20^2
    }));
    return out;
}

TORCH_LIBRARY(my_interp, m) {
    m.def("run(Tensor input, Tensor coords) -> Tensor");
    m.impl("run", TORCH_FN(interp_radius20_launcher));
}
