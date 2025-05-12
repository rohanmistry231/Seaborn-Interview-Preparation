import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)

# Matplotlib scatter plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', alpha=0.5)
plt.title('Matplotlib Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')

# Seaborn scatter plot
plt.subplot(1, 2, 2)
sns.scatterplot(x=x, y=y, color='skyblue', alpha=0.5)
plt.title('Seaborn Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')

# Adjust layout and save
plt.tight_layout()
plt.savefig('seaborn_vs_matplotlib.png')