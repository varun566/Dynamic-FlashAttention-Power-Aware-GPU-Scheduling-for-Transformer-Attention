import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

baseline = json.load(open("../results/baseline_results.json"))
flash = json.load(open("../results/flash_cuda_results.json"))
custom = json.load(open("../results/custom_cuda_results.json"))

def extract(data, L):
    return next(d for d in data if d["L"] == L)

labels = ["Baseline", "FlashAttention CUDA", "Custom CUDA"]

def compute_energy(d):
    return d["avg_power_watts"] * d["avg_latency_ms"] / 1000

# Collect values
L_vals = [512, 1024]

energy = {L: {} for L in L_vals}
power = {L: {} for L in L_vals}
speedup = {L: {} for L in L_vals}

for L in L_vals:
    b = extract(baseline, L)
    f = extract(flash, L)
    c = extract(custom, L)

    # Energy
    energy[L] = {
        "Baseline": compute_energy(b),
        "FlashAttention CUDA": compute_energy(f),
        "Custom CUDA": compute_energy(c),
    }

    # Power
    power[L] = {
        "Baseline": b["avg_power_watts"],
        "FlashAttention CUDA": f["avg_power_watts"],
        "Custom CUDA": c["avg_power_watts"],
    }

    # Speedup
    speedup[L] = {
        "Baseline→Custom CUDA": b["avg_latency_ms"] / c["avg_latency_ms"]
    }

# Build PDF figure
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 1, hspace=0.5)

# Panel 1: Energy
ax1 = fig.add_subplot(gs[0])
for L in L_vals:
    ax1.plot(labels, [energy[L][k] for k in labels], marker="o", label=f"L={L}")
ax1.set_ylabel("Energy (J)")
ax1.set_title("Energy Comparison")
ax1.legend()
ax1.grid(alpha=0.3)

# Panel 2: Power
ax2 = fig.add_subplot(gs[1])
for L in L_vals:
    ax2.plot(labels, [power[L][k] for k in labels], marker="o", label=f"L={L}")
ax2.set_ylabel("Avg Power (W)")
ax2.set_title("Power Consumption Comparison")
ax2.legend()
ax2.grid(alpha=0.3)

# Panel 3: Speedup
ax3 = fig.add_subplot(gs[2])
for L in L_vals:
    ax3.bar([f"L{L}"], [speedup[L]["Baseline→Custom CUDA"]])
ax3.set_ylabel("Speedup (×)")
ax3.set_title("Speedup Over Baseline")
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("combined_report_figure.pdf")
print("Saved: combined_report_figure.pdf")
