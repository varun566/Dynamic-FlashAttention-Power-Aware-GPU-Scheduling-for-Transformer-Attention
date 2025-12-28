import json
import matplotlib.pyplot as plt

# Load results
with open("../results/baseline_results.json") as f:
    baseline = json.load(f)
with open("../results/flash_cuda_results.json") as f:
    flash = json.load(f)
with open("../results/custom_cuda_results.json") as f:
    custom = json.load(f)

def extract(data, L):
    for d in data:
        if d["L"] == L:
            return d
    raise ValueError(f"No entry for L={L}")

labels = ["Baseline", "FlashAttention CUDA", "Custom CUDA"]

energies = {}
for L in [512, 1024]:
    b = extract(baseline, L)
    f_ = extract(flash, L)
    c = extract(custom, L)

    energies[L] = {
        "Baseline": b["avg_power_watts"] * b["avg_latency_ms"] / 1000,
        "FlashAttention CUDA": f_["avg_power_watts"] * f_["avg_latency_ms"] / 1000,
        "Custom CUDA": c["avg_power_watts"] * c["avg_latency_ms"] / 1000,
    }

# Plot
plt.figure(figsize=(10,6))

x = range(len(labels))
bar_w = 0.35

e512 = [energies[512][lab] for lab in labels]
e1024 = [energies[1024][lab] for lab in labels]

plt.bar([i - bar_w/2 for i in x], e512, width=bar_w, label="L=512")
plt.bar([i + bar_w/2 for i in x], e1024, width=bar_w, label="L=1024")

plt.xticks(x, labels, rotation=10)
plt.ylabel("Energy (Joules)")
plt.title("Energy Consumption per Kernel Run")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("energy_plot.png")
print("Saved: energy_plot.png")
