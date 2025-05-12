import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data with exponential growth
np.random.seed(42)
x = np.linspace(1, 100, 100)
y = np.exp(x / 10) + np.random.normal(0, 10, 100)

# Create a scatter plot using Seaborn
sns.scatterplot(x=x, y=y, color='skyblue')

# Apply logarithmic scale to the y-axis
plt.yscale('log')

# Add labels and title
plt.title('Logarithmic Scales: Scatter Plot with Log Y-Axis')
plt.xlabel('X')
plt.ylabel('Y (Log Scale)')

# Save the plot
plt.savefig('logarithmic_scales.png')