import json
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = "../results"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# Load results
baseline = load_json(os.path.join(RESULTS_DIR, "baseline_results.json"))
flash = load_json(os.path.join(RESULTS_DIR, "flash_cuda_results.json"))
custom = load_json(os.path.join(RESULTS_DIR, "custom_cuda_results.json"))

# Extract structured lists
seq_lengths = []
baseline_lat = []
flash_lat = []
custom_lat = []

baseline_power = []
flash_power = []
custom_power = []

for entry in baseline:
    L = entry["L"]
    seq_lengths.append(L)
    baseline_lat.append(entry["avg_latency_ms"])
    baseline_power.append(entry["avg_power_watts"])

# match entries by L for flash and custom
flash_map = {d["L"]: d for d in flash}
custom_map = {d["L"]: d for d in custom}

for L in seq_lengths:
    flash_lat.append(flash_map[L]["avg_latency_ms"])
    custom_lat.append(custom_map[L]["avg_latency_ms"])

    flash_power.append(flash_map[L]["avg_power_watts"])
    custom_power.append(custom_map[L]["avg_power_watts"])

# ------------------------------
# LATENCY PLOT
# ------------------------------
plt.figure(figsize=(8,5))
plt.plot(seq_lengths, baseline_lat, marker='o', label="Baseline")
plt.plot(seq_lengths, flash_lat, marker='o', label="FlashAttention")
plt.plot(seq_lengths, custom_lat, marker='o', label="Custom CUDA")

plt.title("Latency vs Sequence Length")
plt.xlabel("Sequence Length (L)")
plt.ylabel("Avg Latency (ms)")
plt.grid(True)
plt.legend()
plt.savefig("latency_plot.png", dpi=300)
plt.close()

# ------------------------------
# POWER PLOT
# ------------------------------
plt.figure(figsize=(8,5))
plt.plot(seq_lengths, baseline_power, marker='o', label="Baseline")
plt.plot(seq_lengths, flash_power, marker='o', label="FlashAttention")
plt.plot(seq_lengths, custom_power, marker='o', label="Custom CUDA")

plt.title("Power vs Sequence Length")
plt.xlabel("Sequence Length (L)")
plt.ylabel("Avg Power (Watts)")
plt.grid(True)
plt.legend()
plt.savefig("power_plot.png", dpi=300)
plt.close()

# ------------------------------
# SPEEDUP PLOT
# ------------------------------
baseline_lat_arr = np.array(baseline_lat)
flash_lat_arr = np.array(flash_lat)
custom_lat_arr = np.array(custom_lat)

flash_speedup = baseline_lat_arr / flash_lat_arr
custom_speedup = baseline_lat_arr / custom_lat_arr

plt.figure(figsize=(8,5))
plt.plot(seq_lengths, flash_speedup, marker='o', label="FlashAttention Speedup")
plt.plot(seq_lengths, custom_speedup, marker='o', label="Custom CUDA Speedup")

plt.title("Speedup vs Sequence Length")
plt.xlabel("Sequence Length (L)")
plt.ylabel("Speedup (×)")
plt.grid(True)
plt.legend()
plt.savefig("speedup_plot.png", dpi=300)
plt.close()

print("Plots generated: latency_plot.png, power_plot.png, speedup_plot.png")
