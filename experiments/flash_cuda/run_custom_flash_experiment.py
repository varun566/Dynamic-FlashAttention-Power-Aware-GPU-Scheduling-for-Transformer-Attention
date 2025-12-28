import torch
import time
import json
import pynvml
import flash_custom

def measure_power(duration_sec=0.5, interval_sec=0.01):
    """Collect GPU power usage during the given interval."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    samples = []
    t_end = time.time() + duration_sec

    while time.time() < t_end:
        power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)
        samples.append(power_mW / 1000.0)
        time.sleep(interval_sec)

    pynvml.nvmlShutdown()

    if len(samples) == 0:
        return {
            "avg_power": 0,
            "max_power": 0
        }

    return {
        "avg_power": sum(samples) / len(samples),
        "max_power": max(samples)
    }


def run_custom_kernel(L, B, D, H):
    print(f"[CUSTOM CUDA] L={L}, B={B}, d_model={D}, H={H}")

    torch.manual_seed(0)

    Q = torch.randn(B, H, L, D // H, device="cuda", dtype=torch.half)
    K = torch.randn(B, H, L, D // H, device="cuda", dtype=torch.half)
    V = torch.randn(B, H, L, D // H, device="cuda", dtype=torch.half)

    # Warm-up run
    _ = flash_custom.forward(Q, K, V)
    torch.cuda.synchronize()

    iterations = 20
    latencies = []

    start_power_sampling = time.time()
    for _ in range(iterations):
        t0 = time.time()
        _ = flash_custom.forward(Q, K, V)
        torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000)

    # Power measurement for ~0.5 sec
    power_info = measure_power(0.5)

    avg_latency = sum(latencies) / len(latencies)

    result = {
        "L": L,
        "B": B,
        "D": D,
        "H": H,
        "avg_latency_ms": avg_latency,
        "avg_power_watts": power_info["avg_power"],
        "max_power_watts": power_info["max_power"],
    }

    print("[CUSTOM RESULT]")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


if __name__ == "__main__":

    configs = [
        (512, 16, 512, 8),
        (1024, 16, 512, 8),
    ]

    results = []

    for (L, B, D, H) in configs:
        res = run_custom_kernel(L, B, D, H)
        results.append(res)

    # Save results
    out_path = "../results/custom_cuda_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved custom CUDA results to {out_path}")
