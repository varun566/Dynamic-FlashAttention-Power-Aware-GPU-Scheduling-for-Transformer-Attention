#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using at::Tensor;

// Declare the launcher implemented in flash_custom_kernel.cu
extern "C" void flash_custom_forward_launcher(const Tensor& Q,
                                              const Tensor& K,
                                              const Tensor& V,
                                              Tensor& Out);

// Python-visible function
Tensor flash_custom_forward(Tensor Q, Tensor K, Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");

    int B  = Q.size(0);
    int L  = Q.size(1);
    int H  = Q.size(2);
    int Dh = Q.size(3);

    Tensor Out = torch::zeros_like(Q);

    // Call CUDA launcher (this triggers <<< >>> inside the .cu file)
    flash_custom_forward_launcher(Q, K, V, Out);

    return Out;
}

// Bind forward() to Python
PYBIND11_MODULE(flash_custom_ext, m) {
    m.def("forward", &flash_custom_forward, "Custom FlashAttention CUDA forward");
}
