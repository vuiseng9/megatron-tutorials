import matplotlib.pyplot as plt

# Data
x = [1, 2, 4, 8]
ddp = [78.5, 158, 316, 634]
zero2 = [78.5, 150, 270, 516]

ddp_labels = ["78.5 (15)", "158 (30)", "316 (60)", "634 (120)"]
zero2_labels = ["78.5 (15)", "150 (32)", "270 (64)", "516 (128)"]

# Plot
plt.figure(figsize=(10,6))
plt.plot(x, ddp, marker='o', markersize=10, linewidth=3, label="DDP")
plt.plot(x, zero2, marker='o', markersize=10, linewidth=3, label="w/ ZeRO2")
plt.plot([1, 8], [80, 640], linestyle=':', linewidth=3, label="linear growth")

# Annotate
for i, (xi, yi) in enumerate(zip(x, ddp)):
    plt.text(xi, yi, ddp_labels[i], ha='center', va='bottom', fontsize=14)
for i, (xi, yi) in enumerate(zip(x, zero2)):
    plt.text(xi, yi, zero2_labels[i], ha='center', va='bottom', fontsize=14)

plt.xlabel("DP degree", fontsize=18)
plt.ylabel("Total mem. (GBS)", fontsize=18)
plt.title("Total Memory vs DP Degree", fontsize=20)
plt.xticks(x, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(alpha=0.4)

# Save PNG
filename = "/mnt/data/memory_vs_dp.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")

filename

