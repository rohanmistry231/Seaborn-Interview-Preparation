import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'X': np.random.uniform(0, 10, 100),
    'Y': np.random.uniform(0, 20, 100),
    'Category': np.random.choice(['A', 'B', 'C'], 100)
})

# Create a scatter plot with a legend
sns.scatterplot(x='X', y='Y', hue='Category', data=data)

# Customize the legend
plt.legend(title='Group', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

# Add labels and title
plt.title('Customizing Plot Legends: Scatter Plot with Legend')
plt.xlabel('X')
plt.ylabel('Y')

# Adjust layout to prevent legend clipping
plt.tight_layout()

# Save the plot
plt.savefig('customizing_plot_legends.png')