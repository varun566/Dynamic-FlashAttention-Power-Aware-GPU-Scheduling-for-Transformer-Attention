import json
import matplotlib.pyplot as plt
import numpy as np

def load(path):
    with open(path) as f:
        return json.load(f)

baseline = load("../results/baseline_results.json")
flash    = load("../results/flash_cuda_results.json")
custom   = load("../results/custom_cuda_results.json")

def extract_by_L(results, L):
    for r in results:
        if r["L"] == L:
            return r
    return None

Ls = [512, 1024]
labels = ["Baseline", "FlashAttention CUDA", "Custom CUDA"]

energies = {512: [], 1024: []}

for L in Ls:
    b = extract_by_L(baseline, L)
    f = extract_by_L(flash, L)
    c = extract_by_L(custom, L)

    def energy(x):
        return x["avg_power_watts"] * (x["avg_latency_ms"] / 1000)

    energies[L] = [energy(b), energy(f), energy(c)]

x = np.arange(len(labels))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14,5))
for i, L in enumerate(Ls):
    ax = axes[i]
    ax.bar(labels, energies[L], color=["#d9534f", "#0275d8", "#5cb85c"])
    ax.set_title(f"Energy per Run (L = {L})")
    ax.set_ylabel("Energy (Joules)")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Energy Consumption Across Kernels")
plt.tight_layout()
plt.savefig("energy_bar_chart.png", dpi=300)
print("Saved energy_bar_chart.png")
