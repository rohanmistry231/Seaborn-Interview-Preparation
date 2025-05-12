import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)

# Create a JointGrid for custom joint plots
g = sns.JointGrid(x=x, y=y)
g.plot_joint(sns.scatterplot, color='skyblue')
g.plot_marginals(sns.histplot, color='skyblue')

# Add title
plt.suptitle('JointGrid: Custom Joint Plot', y=1.02)

# Save the plot
plt.savefig('jointgrid.png')