import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data (correlation matrix)
np.random.seed(42)
data = np.random.rand(5, 5)
correlation_matrix = np.corrcoef(data)

# Create a heatmap using Seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Add labels and title
plt.title('Heatmap (heatmap): Correlation Matrix')
plt.xlabel('Variable')
plt.ylabel('Variable')

# Save the plot
plt.savefig('heatmap.png')