import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Sample Data (based on the image)
categories = ['Time Saved (hours)', 'Efficiency Gain (x)', 'Cost Savings (%)', 'Scalability Factor']
values = [50.0, 0.0, 100.0, 10.0] # Note: Efficiency Gain is 0 based on the image
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create the bar chart
fig, ax = plt.subplots()
bars = ax.bar(categories, values, color=colors, alpha=0.8)

# Customize the chart
ax.set_title('Cost-Benefit Analysis', fontweight='bold')
ax.set_ylabel('Value')
ax.set_ylim(0, 105) # Set y-axis limit to accommodate the highest value and labels
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability

# Add data labels on top of the bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
            f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# Show the chart
plt.tight_layout() # Adjust layout to prevent overlapping
plt.savefig('Figure_2.png', dpi=300)
print("Figure saved to Figure_2.png")