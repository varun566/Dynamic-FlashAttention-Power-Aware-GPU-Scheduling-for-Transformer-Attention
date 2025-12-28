import json
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Load results
# -------------------------------------------------------------------
with open("../results/baseline_results.json") as f:
    baseline = json.load(f)

with open("../results/flash_cuda_results.json") as f:
    flash = json.load(f)

with open("../results/custom_cuda_results.json") as f:
    custom = json.load(f)

# We'll build synthetic curves for L = 1024 only (can extend later)
def extract(data, L):
    for d in data:
        if d["L"] == L:
            return d
    raise ValueError("No entry found for L", L)

L = 1024
b = extract(baseline, L)
f = extract(flash, L)
c = extract(custom, L)

# -------------------------------------------------------------------
# Generate synthetic power curves
# -------------------------------------------------------------------
def synthetic_curve(avg_power, max_power, duration_ms, resolution=200):
    """
    Creates a smooth, realistic power curve using a sinusoidal bump
    that peaks at max_power.
    """
    t = np.linspace(0, duration_ms, resolution)
    
    # Sinusoidal bump shape — common in GPU workloads
    curve = avg_power + (max_power - avg_power) * np.sin(np.pi * t / duration_ms)**2

    # Add light random noise to simulate sensor jitter
    noise = np.random.normal(0, (max_power - avg_power) * 0.03, size=resolution)
    curve = np.clip(curve + noise, 0, None)

    return t, curve

# Build the curves
tb, pb = synthetic_curve(b["avg_power_watts"], b["max_power_watts"], b["avg_latency_ms"])
tf, pf = synthetic_curve(f["avg_power_watts"], f["max_power_watts"], f["avg_latency_ms"])
tc, pc = synthetic_curve(c["avg_power_watts"], c["max_power_watts"], c["avg_latency_ms"])

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(tb, pb, label=f"Baseline (L={L})", linewidth=2)
plt.plot(tf, pf, label=f"FlashAttention CUDA (L={L})", linewidth=2)
plt.plot(tc, pc, label=f"Custom CUDA (L={L})", linewidth=2)

plt.title(f"Synthetic Power–Time Curve (L = {L})", fontsize=16)
plt.xlabel("Time (ms)", fontsize=14)
plt.ylabel("Power (Watts)", fontsize=14)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig("power_time_curve.png")
print("Saved plot: power_time_curve.png")
