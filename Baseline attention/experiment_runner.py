# experiment_runner.py
import torch
from baseline_attention import benchmark_baseline
from sdpa_attention import benchmark_sdpa
from power_monitor import PowerMonitor


def run_one_config(
    mode: str,
    seq_len: int,
    batch_size: int,
    d_model: int,
    num_heads: int,
    dtype=torch.float16,
    device="cuda",
    warmup=10,
    iters=50,
    power_interval_ms=10.0,
):
    """
    mode: "baseline" or "sdpa"
    """
    assert mode in ("baseline", "sdpa")

    # Prepare NVML power monitor
    pm = PowerMonitor(device_index=0, interval_ms=power_interval_ms)

    # Start power logging
    pm.start()

    # Run benchmark
    if mode == "baseline":
        avg_ms = benchmark_baseline(
            seq_len=seq_len,
            batch_size=batch_size,
            d_model=d_model,
            num_heads=num_heads,
            dtype=dtype,
            device=device,
            warmup=warmup,
            iters=iters,
        )
    else:
        avg_ms = benchmark_sdpa(
            seq_len=seq_len,
            batch_size=batch_size,
            d_model=d_model,
            num_heads=num_heads,
            dtype=dtype,
            device=device,
            warmup=warmup,
            iters=iters,
        )

    # Stop power logging
    pm.stop()
    summary = pm.summary()

    print(f"\n[{mode.upper()} RESULT]")
    print(f"  Avg latency: {avg_ms:.3f} ms")
    if summary is not None:
        print(f"  Avg power:   {summary['avg_power_w']:.2f} W")
        print(f"  Max power:   {summary['max_power_w']:.2f} W")
        print(f"  Energy:      {summary['energy_j']:.3f} J (approx)")
        print(f"  Duration:    {summary['duration_s']:.3f} s")
    else:
        print("  No power samples collected.")

    return avg_ms, summary


if __name__ == "__main__":
    # Example configs – you can turn this into loops over L, B, d_model, etc.
    configs = [
        ("baseline", 512, 16, 512, 8),
        ("sdpa", 512, 16, 512, 8),
        ("baseline", 1024, 16, 512, 8),
        ("sdpa", 1024, 16, 512, 8),
    ]

    for mode, L, B, D, H in configs:
        print("=" * 80)
        print(f"Running mode={mode}, L={L}, B={B}, D={D}, H={H}")
        run_one_config(
            mode=mode,
            seq_len=L,
            batch_size=B,
            d_model=D,
            num_heads=H,
            dtype=torch.float16,
            device="cuda",
            warmup=10,
            iters=50,
            power_interval_ms=10.0,
        )
