import json
import matplotlib.pyplot as plt
import numpy as np

# Helper to load JSON
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# Load data
baseline = load_json("../results/baseline_results.json")
flash = load_json("../results/flash_cuda_results.json")
custom = load_json("../results/custom_cuda_results.json")

seq_lengths = [512, 1024]
kernels = ["Baseline CUDA", "FlashAttention CUDA", "Custom CUDA"]

max_power = {
    "Baseline CUDA": [],
    "FlashAttention CUDA": [],
    "Custom CUDA": []
}

# Populate arrays
for L in seq_lengths:
    # Baseline
    for entry in baseline:
        if entry["L"] == L:
            max_power["Baseline CUDA"].append(entry["max_power_watts"])

    # FlashAttention
    for entry in flash:
        if entry["L"] == L:
            max_power["FlashAttention CUDA"].append(entry["max_power_watts"])

    # Custom CUDA
    # only append the FIRST entry for each L (ignore duplicates)
    seen_L = set()
    for entry in custom:
        if entry["L"] == L and L not in seen_L:
            max_power["Custom CUDA"].append(entry["max_power_watts"])
            seen_L.add(L)

# Plot
x = np.arange(len(seq_lengths))
bar_width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(x - bar_width, max_power["Baseline CUDA"], width=bar_width, label="Baseline CUDA")
plt.bar(x, max_power["FlashAttention CUDA"], width=bar_width, label="FlashAttention CUDA")
plt.bar(x + bar_width, max_power["Custom CUDA"], width=bar_width, label="Custom CUDA Kernel")

plt.xticks(x, seq_lengths)
plt.xlabel("Sequence Length (L)")
plt.ylabel("Max Power (W)")
plt.title("Max Power Consumption Comparison")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("max_power_comparison.png", dpi=300)
print("Generated max_power_comparison.png")
