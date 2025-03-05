import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# Set up argument parser
parser = argparse.ArgumentParser(description='Plot performance comparison and save figure with veclen and rpb info')
parser.add_argument('--exec_name', type=str, default='mini_app_omp',
                   help='Executable name')
parser.add_argument('--veclen', type=int, default=128,
                   help='Vector length')
parser.add_argument('--rpb', type=int, default=2,
                   help='Rows per block')
parser.add_argument('--csv_name', type=str, default='performance_table_mini_apps_H2_burke.csv',
                   help='CSV filename')

args = parser.parse_args()

# Extract mechanism name from csv filename
mechanism_name = args.csv_name.split('apps_')[-1].split('.csv')[0]
figures_dir = f"figures_{mechanism_name}"

# Create figures directory if it doesn't exist
os.makedirs(figures_dir, exist_ok=True)

# Load CSV file
df = pd.read_csv(args.csv_name)

# Update filter criteria with command line arguments
filter_criteria = {
    "exec_name": args.exec_name,
    "veclen": args.veclen, 
    "rpb": args.rpb
}

# Apply filtering
filtered_df = df[
    (df["exec_name"] == filter_criteria["exec_name"]) &
    (df["veclen"] == filter_criteria["veclen"]) &
    (df["rpb"] == filter_criteria["rpb"])
].copy()

filtered_df.loc[:,'cpu_cost (SU)'] = filtered_df['cpu_cost (s)'] / 3600
filtered_df.loc[:,'gpu_cost (SU)'] = filtered_df['gpu_cost (s)'] / 3600 * 4.0
filtered_df.loc[:,'speedup_in_SU'] = filtered_df['cpu_cost (SU)'] / filtered_df['gpu_cost (SU)']
filtered_df.loc[:,'cpu_cost_per_grid (SU)'] = filtered_df['cpu_cost (SU)'] / filtered_df['ng']
filtered_df.loc[:,'gpu_cost_per_grid (SU)'] = filtered_df['gpu_cost (SU)'] / filtered_df['ng']




filtered_df = filtered_df.sort_values(by="ng", ascending=True)
#plt.style.use('seaborn-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300
})


print(filtered_df)
fig, ax1 = plt.subplots(figsize=(10, 6))


bar_width = 0.4
x_labels = filtered_df["ng"].astype(str)  # Convert ng to string for plotting
x = range(len(x_labels))

bar1 = ax1.bar([i - bar_width / 2 for i in x], filtered_df["cpu_cost_per_grid (SU)"], width=bar_width, label="CPU Cost (SU)", color="blue", alpha=0.6)
bar2 = ax1.bar([i + bar_width / 2 for i in x], filtered_df["gpu_cost_per_grid (SU)"], width=bar_width, label="GPU Cost (SU)", color="green", alpha=0.6)

ax1.set_xlabel("ng")
ax1.set_ylabel("Cost/ng (SU)")
ax1.set_xticks(x)  # Changed to place ticks at integer positions
ax1.set_xticklabels(x_labels)
#ax1.legend(loc="upper left")

ax2 = ax1.twinx()
line, = ax2.plot(x, filtered_df["speedup_in_SU"], marker="o", color="red", label="Speedup", linestyle="dashed", linewidth=2)
ax2.set_ylabel("Speedup")
ax2.spines['right'].set_color('red')
ax2.tick_params(axis='y', colors='red')
ax2.yaxis.label.set_color('red')


#ax2.legend(loc="upper right")
# Combine legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + [line], labels1 + labels2, loc="right")


plt.title(f"{filter_criteria['exec_name']}, veclen={filter_criteria['veclen']}, rpb={filter_criteria['rpb']}")
plt.savefig(f"./{figures_dir}/speedupVSprobsize_omp_ng_veclen{filter_criteria['veclen']}_rpb{filter_criteria['rpb']}.png", dpi=300)
