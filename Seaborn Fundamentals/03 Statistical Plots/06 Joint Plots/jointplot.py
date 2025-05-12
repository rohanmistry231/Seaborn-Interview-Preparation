import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)

# Create a joint plot using Seaborn
sns.jointplot(x=x, y=y, kind='scatter', color='skyblue')

# Add title
plt.suptitle('Joint Plot (jointplot): Scatter with Marginal Distributions', y=1.02)

# Save the plot
plt.savefig('jointplot.png')