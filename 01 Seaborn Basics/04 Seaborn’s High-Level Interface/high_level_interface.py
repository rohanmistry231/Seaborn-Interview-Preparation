import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a synthetic dataset
data = pd.DataFrame({
    'X': [1, 2, 3, 4, 5, 6],
    'Y': [2, 4, 5, 4, 5, 6],
    'Category': ['A', 'A', 'B', 'B', 'C', 'C']
})

# Use Seaborn's high-level interface to create a scatter plot with hue
sns.scatterplot(x='X', y='Y', hue='Category', data=data)

# Add labels and title
plt.title('Seaborn High-Level Interface: Scatter Plot with Hue')
plt.xlabel('X')
plt.ylabel('Y')

# Save the plot
plt.savefig('high_level_interface.png')