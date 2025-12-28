import torch
import time
import json
import pynvml


def measure_power(duration_sec=0.1):
    """Measure average GPU power consumption during a short interval."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    start = time.time()
    samples = []

    while time.time() - start < duration_sec:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        samples.append(power_mw / 1000.0)  # convert to Watts
        time.sleep(0.01)

    pynvml.nvmlShutdown()

    return sum(samples) / len(samples), max(samples)


def run_baseline(L=512, B=16, D=512, H=8, warmup=10, iters=50):
    print(f"[BASELINE] L={L}, B={B}, d_model={D}, H={H}")

    device = "cuda"
    D_head = D // H

    q = torch.randn(B, H, L, D_head, device=device)
    k = torch.randn(B, H, L, D_head, device=device)
    v = torch.randn(B, H, L, D_head, device=device)

    # warmup
    for _ in range(warmup):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.cuda.synchronize()

    # timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []

    for _ in range(iters):
        start_event.record()
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))  # ms

    avg_latency = sum(latencies) / len(latencies)

    # measure GPU power
    avg_power, max_power = measure_power(0.2)

    return {
        "L": L,
        "B": B,
        "D": D,
        "H": H,
        "avg_latency_ms": avg_latency,
        "avg_power_watts": avg_power,
        "max_power_watts": max_power,
    }


if __name__ == "__main__":
    sizes = [(512, 16, 512, 8), (1024, 16, 512, 8)]

    results = []

    for (L, B, D, H) in sizes:
        res = run_baseline(L, B, D, H)
        results.append(res)

        print("\n[BASELINE RESULT]")
        for k, v in res.items():
            print(f"  {k}: {v}")

    # save output
    with open("../results/baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved baseline results to results/baseline_results.json")
