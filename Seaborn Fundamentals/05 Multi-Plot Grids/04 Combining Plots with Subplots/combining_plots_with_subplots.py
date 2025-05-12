import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], 100),
    'Value': np.random.normal(0, 1, 100),
    'X': np.random.uniform(0, 10, 100),
    'Y': np.random.uniform(0, 20, 100)
})

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First subplot: Box plot
sns.boxplot(x='Category', y='Value', data=data, color='skyblue', ax=ax1)
ax1.set_title('Box Plot by Category')
ax1.set_xlabel('Category')
ax1.set_ylabel('Value')

# Second subplot: Scatter plot
sns.scatterplot(x='X', y='Y', data=data, color='skyblue', ax=ax2)
ax2.set_title('Scatter Plot')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

# Adjust layout
plt.tight_layout()

# Add a super title
plt.suptitle('Combining Plots with Subplots: Box and Scatter Plots', y=1.05)

# Save the plot
plt.savefig('combining_plots_with_subplots.png')