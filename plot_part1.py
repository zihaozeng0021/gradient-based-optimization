import matplotlib.pyplot as plt
import numpy as np

# Set global font sizes
plt.rcParams.update({
    'font.size': 22,           # Base font size
    'axes.titlesize': 24,      # Title font size
    'axes.labelsize': 22,      # Axis label font size
    'xtick.labelsize': 20,     # X tick label size
    'ytick.labelsize': 20,     # Y tick label size
    'legend.fontsize': 20,     # Legend font size
    'figure.titlesize': 24     # Figure title size
})

# ---------------------------
# Combined Training Plot
# ---------------------------
data_path = "data/part1/training_stats.dat"
data = np.loadtxt(data_path)

epochs = data[:, 0]
iterations = data[:, 1]
loss = data[:, 2]
accuracy = data[:, 3]

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel("Iterations")
ax1.set_ylabel("Test Accuracy", color='tab:blue')
line1, = ax1.plot(iterations, accuracy, marker='o', linestyle='-', color='tab:blue', label="Test Accuracy")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Mean Loss", color='tab:red')
line2, = ax2.plot(iterations, loss, marker='s', linestyle='-', color='tab:red', label="Mean Loss")
ax2.tick_params(axis='y', labelcolor='tab:red')

# Legend inside the plot, middle right
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='center right')

plt.title("Training Performance: Accuracy and Loss vs Iterations")
plt.grid(True)
fig.tight_layout()
plt.savefig("data/part1/combined_training_plot.png")
plt.close()



# ---------------------------
# Accuracy Plot after 30000 Iterations
# ---------------------------
mask = iterations > 30000
iterations_post_30000 = iterations[mask]
accuracy_post_30000 = accuracy[mask]

plt.figure(figsize=(10, 6))
plt.plot(iterations_post_30000, accuracy_post_30000, marker='o', linestyle='-', label="Test Accuracy > 30000")
plt.xlabel("Iterations")
plt.ylabel("Test Accuracy")
plt.ylim(0.9, 1.0)
plt.title("Test Accuracy (Iterations > 30000)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("data/part1/accuracy_after30000.png")
plt.close()

# ---------------------------
# Validation Comparison Plot
# ---------------------------
val_file = "data/part1/validation_stats.dat"
val_data = []
with open(val_file, "r") as f:
    lines = f.readlines()
    for line in lines:
        numbers = [float(x) for x in line.strip().split()]
        val_data.append(numbers)

forward_avg = np.mean(val_data[0]) if len(val_data) > 0 and len(val_data[0]) > 0 else 0.0
backward_avg = np.mean(val_data[1]) if len(val_data) > 1 and len(val_data[1]) > 0 else 0.0
central_avg = np.mean(val_data[2]) if len(val_data) > 2 and len(val_data[2]) > 0 else 0.0

methods = ["Forward", "Backward", "Central"]
averages = [forward_avg, backward_avg, central_avg]

plt.figure(figsize=(8, 6))
plt.bar(methods, averages, color=['tab:blue', 'tab:green', 'tab:red'])
plt.ylabel("Average Relative Difference (%)")
plt.title("Average Validation Differences")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/part1/validation_comparison.png")
plt.close()
