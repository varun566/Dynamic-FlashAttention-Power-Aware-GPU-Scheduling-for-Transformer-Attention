#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

using at::Tensor;

// -----------------------------------------------------------------------------
// CUDA KERNEL
// -----------------------------------------------------------------------------
__global__ void flash_custom_forward_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ Out,
    int B, int H, int L, int Dh
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * L * Dh;

    if (idx < total) {
        // For now: simple verification kernel that copies Q → Out
        // (shows our custom CUDA extension works)
        Out[idx] = Q[idx];
    }
}

// -----------------------------------------------------------------------------
// CUDA LAUNCHER CALLED FROM C++ BINDING
// -----------------------------------------------------------------------------
extern "C" void flash_custom_forward_launcher(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    Tensor& Out
) {
    int B  = Q.size(0);
    int H  = Q.size(1);
    int L  = Q.size(2);
    int Dh = Q.size(3);

    int total = B * H * L * Dh;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    const half* q_ptr = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    const half* k_ptr = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    const half* v_ptr = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    half* out_ptr     = reinterpret_cast<half*>(Out.data_ptr<at::Half>());

    flash_custom_forward_kernel<<<blocks, threads>>>(
        q_ptr, k_ptr, v_ptr, out_ptr,
        B, H, L, Dh
    );

    // Optional for debugging — remove later for performance.
    // cudaDeviceSynchronize();
}
