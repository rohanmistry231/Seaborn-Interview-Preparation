import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data (correlation matrix)
np.random.seed(42)
data = np.random.rand(5, 5)
correlation_matrix = np.corrcoef(data)

# Create a clustermap using Seaborn
sns.clustermap(correlation_matrix, annot=True, cmap='coolwarm')

# Add title
plt.title('Clustermap (clustermap): Clustered Correlation Matrix')

# Save the plot
plt.savefig('clustermap.png')