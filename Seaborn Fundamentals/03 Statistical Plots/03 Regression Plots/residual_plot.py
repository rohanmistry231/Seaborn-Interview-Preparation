import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data
np.random.seed(42)
x = np.random.uniform(0, 10, 50)
y = 2 * x + np.random.normal(0, 1, 50)

# Create a residual plot using Seaborn
sns.residplot(x=x, y=y, color='skyblue')

# Add labels and title
plt.title('Residual Plot (residplot): Residuals of Regression')
plt.xlabel('X')
plt.ylabel('Residuals')

# Save the plot
plt.savefig('residual_plot.png')