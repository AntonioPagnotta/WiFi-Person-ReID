import matplotlib.pyplot as plt
import numpy as np

# Data definitions
epochs = list(range(1, 21))

# Model 1: Modello 1 - modello originale paper
m1_loss = [1.19461, 0.30714, 0.24202, 0.20164, 0.17837, 0.16005, 0.14680, 0.13568, 0.12685, 0.11833,
           0.11464, 0.10731, 0.10345, 0.10014, 0.09648, 0.09193, 0.09131, 0.08748, 0.08719, 0.08371]
m1_r1 = [64.815, 67.901, 70.370, 66.667, 66.667, 62.963, 62.346, 64.815, 64.198, 61.728,
         56.790, 55.556, 56.790, 59.877, 56.173, 62.346, 56.790, 53.086, 56.790, 56.173]
m1_map = [67.593, 66.049, 64.815, 66.667, 66.667, 68.519, 68.827, 67.593, 67.901, 69.136,
          71.605, 72.222, 71.605, 70.062, 71.914, 68.827, 71.605, 73.457, 71.605, 71.914]

# Model 2: Modello 2 - modello open set + bug fix
m2_loss = [0.28420, 0.07773, 0.06392, 0.05261, 0.03811, 0.03318, 0.03120, 0.02937, 0.02852, 0.02736,
           0.02643, 0.02563, 0.02558, 0.02535, 0.02478, 0.02467, 0.02446, 0.02383, 0.02340, 0.02305]
m2_r1 = [70.732, 62.195, 62.439, 55.122, 55.122, 57.073, 52.927, 51.707, 56.829, 55.366,
         56.341, 55.366, 56.585, 53.415, 50.976, 56.585, 56.585, 54.146, 53.171, 57.073]
m2_map = [55.171, 48.268, 46.820, 40.580, 40.850, 43.030, 39.824, 39.079, 41.473, 39.834,
          40.373, 40.019, 43.240, 40.112, 38.732, 42.004, 43.191, 40.346, 39.055, 41.689]

# Model 3: Modello 3 open set before bug fix
m3_loss = [0.70656, 0.16261, 0.12018, 0.10285, 0.08769, 0.07946, 0.07231, 0.06596, 0.06156, 0.05944,
           0.05651, 0.05470, 0.05216, 0.05294, 0.04994, 0.04716, 0.04624, 0.04730, 0.04591, 0.04423]
m3_r1 = [64.634, 56.341, 57.073, 53.415, 45.366, 43.171, 45.122, 47.317, 38.780, 41.707,
         23.902, 11.463, 10.732, 10.488, 8.049, 8.293, 15.122, 7.317, 9.756, 2.439]
m3_map = [86.147, 87.924, 87.768, 88.552, 90.277, 90.747, 90.329, 89.859, 91.688, 91.061,
          94.877, 97.543, 97.700, 97.752, 98.275, 98.223, 96.759, 98.432, 97.909, 99.477]

# Model 4: Modello 4 open set + HardTripletLoss
m4_loss = [0.47719, 0.49647, 0.47614, 0.45340, 0.44238, 0.44002, 0.42881, 0.42903, 0.41487, 0.40730,
           0.40234, 0.40361, 0.39717, 0.39625, 0.38533, 0.38475, 0.38010, 0.38402, 0.38162, 0.37479]
m4_r1 = [42.195, 41.707, 41.951, 43.171, 43.659, 40.488, 40.976, 40.732, 39.756, 41.951,
         43.171, 45.366, 44.146, 45.854, 46.098, 47.561, 47.805, 47.561, 46.829, 46.829]
m4_map = [26.681, 26.229, 26.772, 27.032, 26.809, 26.601, 26.772, 26.689, 26.400, 26.869,
          27.351, 27.852, 28.382, 29.381, 29.498, 29.570, 29.902, 30.064, 30.098, 29.831]

models_data = [
    ("Original paper model", m1_loss, m1_r1, m1_map),
    ("v_2 - Open set before bug fix", m3_loss, m3_r1, m3_map),
    ("v_2.5 - Open set + bug fix", m2_loss, m2_r1, m2_map),
    ("v_3 - Open set + HardTripletLoss", m4_loss, m4_r1, m4_map)
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (name, loss, r1, mmap) in enumerate(models_data):
    ax = axes[idx]
    # We can plot Loss on left axis and R1/mAP on right axis, or normalize / plot together with dual axes
    ax2 = ax.twinx()

    line1 = ax.plot(epochs, loss, 'r-', label='Loss', marker='o')
    line2 = ax2.plot(epochs, r1, 'b-', label='Rank #1 (%)', marker='s')
    line3 = ax2.plot(epochs, mmap, 'g-', label='mAP (%)', marker='^')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='r')
    ax2.set_ylabel('Metrics (%)', color='b')
    ax.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='b')

    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')

    ax.set_title(name)
    ax.grid(True)

plt.tight_layout()
plt.savefig('individual_models_metrics.png', dpi=300)
plt.show()