# baseline_attention.py
import math
import torch
from torch import nn


class BaselineAttention(nn.Module):
    """
    Naive scaled dot-product attention:
    scores = (Q K^T) / sqrt(d_k)
    attn   = softmax(scores)
    out    = attn V
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        # q, k, v: [B, H, L, D]
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,L,L]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B,H,L,D]
        return out


def benchmark_baseline(
    seq_len=1024,
    batch_size=32,
    d_model=512,
    num_heads=8,
    dtype=torch.float16,
    device="cuda",
    warmup=10,
    iters=50,
):
    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    device = torch.device(device)
    q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=dtype)

    attn = BaselineAttention().to(device=device, dtype=dtype)

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

    total_ms = start_event.elapsed_time(end_event)  # ms over all iters
    avg_ms = total_ms / iters

    print(f"[BASELINE] L={seq_len}, B={batch_size}, d_model={d_model}, H={num_heads}")
    print(f"  Avg latency: {avg_ms:.3f} ms over {iters} iterations")

    return avg_ms
