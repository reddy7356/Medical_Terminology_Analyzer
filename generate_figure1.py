import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Set style for publication quality
plt.rcParams.update({'font.size': 8, 'font.family': 'sans-serif'})

# Create figure with a custom layout to prevent overlap
fig = plt.figure(figsize=(15, 10), dpi=300) # High resolution 300 DPI
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.3, hspace=0.4)

# --- 1. Case Distribution (Pie Chart) ---
ax1 = fig.add_subplot(gs[0, 0])
sizes = [139, 861]
labels = ['Cardio-Resp\n(13.9%)', 'Mixed/Other\n(86.1%)']
colors = ['#2ca02c', '#ff7f0e'] # Green, Orange
wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
ax1.set_title('Case Distribution', fontweight='bold')

# --- 2. Model F1-Scores (Bar Chart) ---
ax2 = fig.add_subplot(gs[0, 1])
models = ['Log Reg', 'Random\nForest', 'SVM', 'Naive\nBayes']
f1_scores = [0.526, 0.571, 0.711, 0.303]
bars = ax2.bar(models, f1_scores, color=['#5da5da', '#faa43a', '#60bd68', '#f15854'])
ax2.set_ylim(0, 1.0)
ax2.set_ylabel('F1-Score')
ax2.set_title('Model F1-Scores', fontweight='bold')
# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.3f}', ha='center', va='bottom', fontsize=7)

# --- 3. Keywords by Category (Horizontal Bar) ---
ax3 = fig.add_subplot(gs[0, 2])
categories = ['Cardio', 'Resp', 'Other']
counts = [17022, 11812, 23545]
y_pos = np.arange(len(categories))
ax3.barh(y_pos, counts, color=['#d9534f', '#428bca', '#f0ad4e'])
ax3.set_yticks(y_pos)
ax3.set_yticklabels(categories)
ax3.invert_yaxis()  # Labels read top-to-bottom
ax3.set_xlabel('Count')
ax3.set_title('Keywords by Category', fontweight='bold')
# Add counts inside/next to bars
for i, v in enumerate(counts):
    ax3.text(v + 500, i, str(v), color='black', va='center', fontweight='bold', fontsize=7)

# --- 4. Processing Time (Log Scale) ---
ax4 = fig.add_subplot(gs[1, 0:2]) # Span 2 columns
times = [3000, 1] # 3000 mins vs ~0 mins (1 for log scale visibility)
labels_time = ['Human (Estimated)', 'Machine (Actual)']
bars4 = ax4.bar(labels_time, times, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
ax4.set_yscale('log')
ax4.set_ylabel('Time (minutes) - Log Scale')
ax4.set_title('Processing Time Comparison', fontweight='bold')
ax4.set_ylim(0.1, 10000)
# Add text labels
ax4.text(0, 3500, '3000.0 min', ha='center', fontweight='bold')
ax4.text(1, 1.5, '< 0.01 min', ha='center', fontweight='bold')

# --- 5. Key Statistics (Text Box) ---
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off') # Hide axes
stats_text = (
    "KEY STATISTICS\n\n"
    "Total Cases: 1,000\n"
    "Cardio-Resp Cases: 139 (13.9%)\n\n"
    "Best Model: SVM\n"
    "F1-Score: 0.711\n"
    "Precision: 0.941\n"
    "Recall: 0.571\n\n"
    "Processing Time:\n"
    "• Human: 3000.0 min\n"
    "• Machine: 0.003 sec/case\n"
    "• Efficiency: 3000x Faster"
)
# Add a rounded box
props = dict(boxstyle='round', facecolor='#e6e6e6', alpha=0.5)
ax5.text(0.05, 0.5, stats_text, fontsize=9, verticalalignment='center', bbox=props)

# Overall Title
plt.suptitle('Cardio-Respiratory Case Classification Analysis', fontsize=14, fontweight='bold', y=0.98)

# Save
output_path = 'Figure_1_HighRes.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Figure saved to {output_path}")