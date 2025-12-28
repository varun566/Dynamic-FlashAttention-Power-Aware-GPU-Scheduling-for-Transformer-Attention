import os
from torch.utils.cpp_extension import load

# Directory where this file lives
_src_dir = os.path.dirname(__file__)

# JIT-compile / load the CUDA extension
_flash_ext = load(
    name="flash_custom_ext",
    sources=[
        os.path.join(_src_dir, "flash_custom_kernel.cu"),
        os.path.join(_src_dir, "flash_custom_binding.cpp"),
    ],
    verbose=True,
)

def forward(Q, K, V):
    """
    Python wrapper around the custom CUDA kernel.

    Q, K, V: [B, L, H, Dh] FP16 CUDA tensors
    returns: Out tensor with same shape as Q
    """
    return _flash_ext.forward(Q, K, V)
