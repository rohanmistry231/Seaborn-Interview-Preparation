import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data
np.random.seed(42)
x = np.random.uniform(0, 10, 50)
y = 2 * x + np.random.normal(0, 1, 50)

# Create a regression plot using Seaborn
sns.regplot(x=x, y=y, color='skyblue')

# Add labels and title
plt.title('Regression Plot (regplot): Scatter with Regression Line')
plt.xlabel('X')
plt.ylabel('Y')

# Save the plot
plt.savefig('regplot.png')