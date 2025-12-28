import json
import matplotlib.pyplot as plt
import numpy as np

# Helper to load JSON
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# Load experiment results
baseline = load_json("../results/baseline_results.json")
flash = load_json("../results/flash_cuda_results.json")
custom = load_json("../results/custom_cuda_results.json")

seq_lengths = [512, 1024]
kernels = ["Baseline CUDA", "FlashAttention CUDA", "Custom CUDA"]

avg_power = {
    "Baseline CUDA": [],
    "FlashAttention CUDA": [],
    "Custom CUDA": []
}

# Populate arrays
for L in seq_lengths:
    # Baseline
    for entry in baseline:
        if entry["L"] == L:
            avg_power["Baseline CUDA"].append(entry["avg_power_watts"])

    # FlashAttention
    for entry in flash:
        if entry["L"] == L:
            avg_power["FlashAttention CUDA"].append(entry["avg_power_watts"])

    # Custom CUDA  — take the first entry only to avoid duplicate L=512
    seen_L = set()
    for entry in custom:
        if entry["L"] == L and L not in seen_L:
            avg_power["Custom CUDA"].append(entry["avg_power_watts"])
            seen_L.add(L)

# Plotting
x = np.arange(len(seq_lengths))
bar_width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(x - bar_width, avg_power["Baseline CUDA"], width=bar_width, label="Baseline CUDA")
plt.bar(x, avg_power["FlashAttention CUDA"], width=bar_width, label="FlashAttention CUDA")
plt.bar(x + bar_width, avg_power["Custom CUDA"], width=bar_width, label="Custom CUDA Kernel")

plt.xticks(x, seq_lengths)
plt.xlabel("Sequence Length (L)")
plt.ylabel("Average Power (W)")
plt.title("Average Power Consumption Comparison")
plt.grid(alpha=0.3)
plt.legend()

plt.savefig("avg_power_comparison.png", dpi=300)
print("Generated avg_power_comparison.png")
