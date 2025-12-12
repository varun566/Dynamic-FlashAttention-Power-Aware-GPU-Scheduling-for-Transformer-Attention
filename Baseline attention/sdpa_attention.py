# sdpa_attention.py
import torch
from torch import nn
import torch.nn.functional as F


class SDPAAttention(nn.Module):
    """
    Wrapper around torch.nn.functional.scaled_dot_product_attention.
    On H100 + torch 2.5.1, this will use fused/flash-style kernels internally.
    """

    def __init__(self, is_causal: bool = False):
        super().__init__()
        self.is_causal = is_causal

    def forward(self, q, k, v, attn_mask=None):
        # q, k, v: [B, H, L, D]
        # attn_mask: [B, 1, L, L] or None
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=self.is_causal
        )


def benchmark_sdpa(
    seq_len=1024,
    batch_size=32,
    d_model=512,
    num_heads=8,
    dtype=torch.float16,
    device="cuda",
    warmup=10,
    iters=50,
    is_causal=False,
):
    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    device = torch.device(device)
    q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=dtype)

    attn = SDPAAttention(is_causal=is_causal).to(device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup):
        _ = attn(q, k, v)
    torch.cuda.synchronize()

    # Timed runs with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        _ = attn(q, k, v)
    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters

    print(f"[SDPA] L={seq_len}, B={batch_size}, d_model={d_model}, H={num_heads}")
    print(f"  Avg latency: {avg_ms:.3f} ms over {iters} iterations")

    return avg_ms
