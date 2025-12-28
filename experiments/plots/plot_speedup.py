import json
import matplotlib.pyplot as plt

baseline = json.load(open("../results/baseline_results.json"))
flash = json.load(open("../results/flash_cuda_results.json"))
custom = json.load(open("../results/custom_cuda_results.json"))

def extract(data, L):
    return next(d for d in data if d["L"] == L)

labels = ["Baseline→FlashAttention", "Baseline→Custom CUDA", "Flash→Custom CUDA"]

speedup = {}

for L in [512, 1024]:
    b = extract(baseline, L)["avg_latency_ms"]
    f = extract(flash, L)["avg_latency_ms"]
    c = extract(custom, L)["avg_latency_ms"]

    speedup[L] = {
        "Baseline→FlashAttention": b / f,
        "Baseline→Custom CUDA": b / c,
        "Flash→Custom CUDA": f / c,
    }

plt.figure(figsize=(10,5))
x = range(len(labels))
bar_w = 0.35

s512 = [speedup[512][lab] for lab in labels]
s1024 = [speedup[1024][lab] for lab in labels]

plt.bar([i - bar_w/2 for i in x], s512, width=bar_w, label="L=512")
plt.bar([i + bar_w/2 for i in x], s1024, width=bar_w, label="L=1024")

plt.xticks(x, labels, rotation=12)
plt.ylabel("Speedup (×)")
plt.title("Kernel Speedup Comparison")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("speedup_plot.png")
print("Saved: speedup_plot.png")
