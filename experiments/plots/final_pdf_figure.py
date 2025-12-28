import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load all generated figures
fig_names = [
    "latency_plot.png",
    "max_power_comparison.png",
    "energy_bar_chart.png",
    "gflops_efficiency.png",
    "power_time_modeled.png"
]

imgs = [mpimg.imread(f) for f in fig_names]

fig, axes = plt.subplots(len(imgs), 1, figsize=(8.27, 11.69))  # A4 portrait

titles = [
    "Latency Comparison",
    "Max Power Comparison",
    "Energy Consumption",
    "GFLOPs/W Efficiency",
    "Modeled Power-Time Curves"
]

for ax, img, title in zip(axes, imgs, titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig("final_report_figure.pdf", dpi=300)
print("Saved final_report_figure.pdf")
