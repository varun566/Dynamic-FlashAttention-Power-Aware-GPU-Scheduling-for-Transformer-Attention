import json
import numpy as np
import matplotlib.pyplot as plt

def load(path):
    with open(path) as f:
        return json.load(f)

baseline = load("../results/baseline_results.json")
flash    = load("../results/flash_cuda_results.json")
custom   = load("../results/custom_cuda_results.json")

def extract(results, L):
    return [r for r in results if r["L"] == L][0]

def compute_flops(B, H, L, D):
    dh = D // H
    return 2 * B * H * (L**2) * dh  # FLOPs

kernels = ["baseline", "flash", "custom"]
data_sources = {"baseline": baseline, "flash": flash, "custom": custom}

Ls = [512, 1024]

fig, axes = plt.subplots(1, 2, figsize=(14,5))

for idx, L in enumerate(Ls):
    ax = axes[idx]

    gflops_eff = []
    for name in kernels:
        entry = extract(data_sources[name], L)
        FLOPs = compute_flops(entry["B"], entry["H"], entry["L"], entry["D"])
        GFLOPs = FLOPs / 1e9
        eff = GFLOPs / entry["avg_power_watts"]  # GFLOPs per Watt
        gflops_eff.append(eff)

    ax.bar(kernels, gflops_eff, color=["#d9534f", "#0275d8", "#5cb85c"])
    ax.set_title(f"GFLOPs/W Efficiency (L = {L})")
    ax.set_ylabel("GFLOPs per Watt")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Energy Efficiency of Kernels (GFLOPs per Watt)")
plt.tight_layout()
plt.savefig("gflops_efficiency.png", dpi=300)
print("Saved gflops_efficiency.png")
