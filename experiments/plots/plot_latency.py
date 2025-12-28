import json
import matplotlib.pyplot as plt
import numpy as np

# Load JSON result files
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

baseline = load_json("../results/baseline_results.json")
flash = load_json("../results/flash_cuda_results.json")
custom = load_json("../results/custom_cuda_results.json")

# Extract latency for L = 512 and 1024
seq_lengths = [512, 1024]
kernels = ["Baseline CUDA", "FlashAttention CUDA", "Custom CUDA"]

latencies = {
    "Baseline CUDA": [],
    "FlashAttention CUDA": [],
    "Custom CUDA": []
}

# Fill latency arrays
for L in seq_lengths:
    # Baseline
    for entry in baseline:
        if entry["L"] == L:
            latencies["Baseline CUDA"].append(entry["avg_latency_ms"])

    # FlashAttention
    for entry in flash:
        if entry["L"] == L:
            latencies["FlashAttention CUDA"].append(entry["avg_latency_ms"])

    # Custom CUDA
    for entry in custom:
        if entry["L"] == L:
            latencies["Custom CUDA"].append(entry["avg_latency_ms"])

# Plotting
x = np.arange(len(seq_lengths))  # [0, 1]
bar_width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(x - bar_width, latencies["Baseline CUDA"], width=bar_width, label="Baseline CUDA")
plt.bar(x, latencies["FlashAttention CUDA"], width=bar_width, label="FlashAttention CUDA")
plt.bar(x + bar_width, latencies["Custom CUDA"], width=bar_width, label="Custom CUDA Kernel")

plt.xticks(x, seq_lengths)
plt.xlabel("Sequence Length (L)")
plt.ylabel("Avg Latency (ms)")
plt.title("Latency Comparison Across Kernel Types")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("latency_comparison.png", dpi=300)
print("Generated latency_comparison.png")
