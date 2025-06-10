import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Set global font sizes for consistency
plt.rcParams.update({
    'font.size': 22,           # Base font size
    'axes.titlesize': 24,      # Title font size
    'axes.labelsize': 22,      # Axis label font size
    'xtick.labelsize': 20,     # X tick label size
    'ytick.labelsize': 20,     # Y tick label size
    'legend.fontsize': 16,     # Legend font size
    'figure.titlesize': 24     # Figure title size
})

# ---------------------------
# Accuracy Plot for Part 2
# ---------------------------
file_pattern = "data/part2/combined/training_stats_*.dat"
files = glob.glob(file_pattern)
files.sort()

plt.figure(figsize=(10, 6))

for file_path in files:
    base_name = os.path.basename(file_path)
    tokens = base_name.replace("_combination.dat", "").split('_')
    if len(tokens) < 4:
        print(f"Skipping file {base_name} due to unexpected naming format.")
        continue

    lr_str = tokens[2]
    bs_str = tokens[3]
    lr = float(lr_str[0] + '.' + lr_str[1:])
    batch = int(bs_str)

    data = np.loadtxt(file_path)
    iterations = data[:, 1]
    accuracy = data[:, 3]
    plt.plot(iterations, accuracy, marker='o', linestyle='-', label=f"lr={lr}, bs={batch}")
plt.xlabel("Iterations")
plt.ylabel("Test Accuracy")
plt.title("Training Accuracy vs Iterations (Combined)")
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("data/part2/combined/accuracy_training_plot_combined.png")
plt.close()

# ---------------------------
# Accuracy Plot after 30000 iterations (y-axis from 0.8 to 1)
# ---------------------------
plt.figure(figsize=(10, 6))

for file_path in files:
    base_name = os.path.basename(file_path)
    tokens = base_name.replace("_combination.dat", "").split('_')
    if len(tokens) < 4:
        continue

    lr_str = tokens[2]
    bs_str = tokens[3]
    lr = float(lr_str[0] + '.' + lr_str[1:])
    batch = int(bs_str)
    data = np.loadtxt(file_path)
    iterations = data[:, 1]
    accuracy = data[:, 3]

    mask = iterations > 30000
    if np.sum(mask) == 0:
        continue

    iterations_post_30000 = iterations[mask]
    accuracy_post_30000 = accuracy[mask]

    plt.plot(iterations_post_30000, accuracy_post_30000, marker='o', linestyle='-', label=f"lr={lr}, bs={batch}")

plt.xlabel("Iterations")
plt.ylabel("Test Accuracy")
plt.ylim(0.85, 1)
plt.title("Test Accuracy (Iterations > 30000) (Combined)")
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("data/part2/combined/accuracy_training_plot_combined_post_30000.png")
plt.close()
