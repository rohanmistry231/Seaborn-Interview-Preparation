import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create a KDE plot using Seaborn
sns.kdeplot(data, color='red', linewidth=2)

# Add labels and title
plt.title('Kernel Density Estimation (kdeplot): Synthetic Data')
plt.xlabel('Value')
plt.ylabel('Density')

# Save the plot
plt.savefig('kdeplot.png')