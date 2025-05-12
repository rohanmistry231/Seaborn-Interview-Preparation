import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create a distribution plot using Seaborn
sns.displot(data, bins=30, color='skyblue', kde=True)

# Add labels and title
plt.title('Distribution Plot (displot): Histogram with KDE')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Save the plot
plt.savefig('displot.png')