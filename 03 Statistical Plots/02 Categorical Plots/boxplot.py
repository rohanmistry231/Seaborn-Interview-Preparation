import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], 100),
    'Value': np.random.normal(0, 1, 100)
})

# Create a box plot using Seaborn
sns.boxplot(x='Category', y='Value', data=data, color='skyblue')

# Add labels and title
plt.title('Box Plot (boxplot): Values by Category')
plt.xlabel('Category')
plt.ylabel('Value')

# Save the plot
plt.savefig('boxplot.png')