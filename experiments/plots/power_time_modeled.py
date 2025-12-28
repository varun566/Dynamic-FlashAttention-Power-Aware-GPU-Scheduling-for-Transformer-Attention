import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
def load_json(path):
    with open(path) as f:
        return json.load(f)

baseline = load_json("../results/baseline_results.json")
flash    = load_json("../results/flash_cuda_results.json")
custom   = load_json("../results/custom_cuda_results.json")

# Helper: generate modeled curve
def generate_curve(avg_power, max_power, time_ms, steps=200):
    t = np.linspace(0, time_ms/1000, steps)   # seconds
    # modeled curve: cosine rise from avg to max
    p = avg_power + (max_power - avg_power) * (0.5 - 0.5*np.cos(np.linspace(0, np.pi, steps)))
    return t, p

# Extract data for L = 512 and L = 1024
def extract(results, L):
    for r in results:
        if r["L"] == L:
            return r
    return None

configs = [
    (baseline, flash, custom, 512),
    (baseline, flash, custom, 1024)
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
titles = ["Sequence Length 512", "Sequence Length 1024"]

for ax, (base, fla, cus, L), title in zip(axes, configs, titles):

    b = extract(base, L)
    f = extract(fla, L)
    c = extract(cus, L)

    # baseline curve
    tb, pb = generate_curve(b["avg_power_watts"], b["max_power_watts"], b["avg_latency_ms"])
    ax.plot(tb, pb, label="Baseline", linewidth=2.5)

    # flash curve
    tf, pf = generate_curve(f["avg_power_watts"], f["max_power_watts"], f["avg_latency_ms"])
    ax.plot(tf, pf, label="FlashAttention CUDA", linewidth=2.5)

    # custom curve
    tc, pc = generate_curve(c["avg_power_watts"], c["max_power_watts"], c["avg_latency_ms"])
    ax.plot(tc, pc, label="Custom CUDA Kernel", linewidth=2.5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Modeled Power–Time Curves for Attention Kernels", fontsize=16)
plt.tight_layout()
plt.savefig("power_time_modeled.png", dpi=300)
print("Saved power_time_modeled.png")
