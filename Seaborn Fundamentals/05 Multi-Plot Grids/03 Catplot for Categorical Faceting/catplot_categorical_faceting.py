import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], 100),
    'Value': np.random.normal(0, 1, 100),
    'Group': np.random.choice(['G1', 'G2'], 100)
})

# Create a catplot for categorical faceting
sns.catplot(x='Category', y='Value', col='Group', kind='box', data=data, color='skyblue')

# Add title
plt.suptitle('Catplot: Categorical Faceting with Box Plots', y=1.05)

# Save the plot
plt.savefig('catplot_categorical_faceting.png')