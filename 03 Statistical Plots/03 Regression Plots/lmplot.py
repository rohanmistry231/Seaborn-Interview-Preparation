import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'X': np.random.uniform(0, 10, 50),
    'Y': np.random.uniform(0, 20, 50),
    'Category': np.random.choice(['A', 'B'], 50)
})

# Create a linear model plot using Seaborn
sns.lmplot(x='X', y='Y', hue='Category', data=data)

# Add title
plt.title('Linear Model Plot (lmplot): Regression by Category')

# Save the plot
plt.savefig('lmplot.png')