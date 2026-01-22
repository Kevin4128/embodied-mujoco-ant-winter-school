#!/usr/bin/env python3
"""
Plot average rewards from SARSA training run.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os
import json

# Path to the CSV file.
csv_path = 'logs/sarsa_ant_forward_20260121_230625/run_SimEmbodiedAnt_average_rewards.csv'
csv_path_logging_data = 'logs/sarsa_ant_forward_20260121_230625/logging_data.csv'
# Read the CSV file.
print(f"Reading data from {csv_path}...")
df = pd.read_csv(csv_path)
df_logging_data = pd.read_csv(csv_path_logging_data)

print(f"Loaded {len(df)} data points")
print(f"Step range: {df['step'].min():.1f} to {df['step'].max():.1f}")
print(f"Reward range: {df['reward'].min():.4f} to {df['reward'].max():.4f}")

# Read the config file.
config_path = os.path.join(os.path.dirname(csv_path), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

DT = config['dt']
# Remove first N.
N = 120
df = df[int(N*DT):]

# Create the plot with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot average reward on first subplot
color1 = 'tab:blue'
ax1.plot(df['step'] * DT, df['reward'], color=color1, linewidth=1.0, label='Average Reward (per step)')
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('Average Reward', color=color1, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=11)
ax1.set_title('Average Reward (per step)', fontsize=12, fontweight='bold')

# Plot return per timelimit on second subplot
color2 = 'tab:orange'
ax2.plot(df_logging_data['timelimit_episode'], df_logging_data['return_per_timelimit'],
         color=color2, linewidth=1.0, label='Return per Timelimit (per episode)')
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Return Per Timelimit', color=color2, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=11)
ax2.set_title('Return per Timelimit (per episode)', fontsize=12, fontweight='bold')

# Overall title
fig.suptitle('SARSA Learning: Average Reward and Return per Timelimit', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the plot.
output_path = os.path.join(os.path.dirname(csv_path), 'average_rewards_plot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.close()
