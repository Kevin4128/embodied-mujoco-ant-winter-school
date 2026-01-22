import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import json

# Path to the CSV file.
folder = 'runs/sim_learning_to_walk_20260121-234020_seed_1'
csv_path = os.path.join(folder, 'SimEmbodiedAnt_average_rewards.csv')
# Read the CSV file.
print(f"Reading data from {csv_path}...")
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} data points")
print(f"Step range: {df['step'].min():.1f} to {df['step'].max():.1f}")
print(f"Reward range: {df['reward'].min():.4f} to {df['reward'].max():.4f}")

# Read the config file.
config_path = os.path.join(folder, 'weights_and_args', 'args.json')
with open(config_path, 'r') as f:
    config = json.load(f)
DT = config['dt']
# Remove first N.
N = 120
df = df[int(N/DT):]

fig, ax = plt.subplots(1, 1)

color = 'tab:blue'
ax.plot(df['step']*DT/3600, df['reward'], color=color, linewidth=1.0, label='Average Reward')
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Average Reward', color=color, fontsize=12)
ax.tick_params(axis='y', labelcolor=color)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=11)
ax.set_title('Average Reward', fontsize=12, fontweight='bold')

fig.suptitle('Average Reward', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = os.path.join(os.path.dirname(csv_path), 'average_rewards_plot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.show()
