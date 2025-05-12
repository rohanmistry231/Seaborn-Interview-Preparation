import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'X': np.random.uniform(0, 10, 100),
    'Y': np.random.uniform(0, 20, 100),
    'Category': np.random.choice(['A', 'B'], 100),
    'Group': np.random.choice(['G1', 'G2'], 100)
})

# Create a FacetGrid plot
g = sns.FacetGrid(data, col='Group', hue='Category')
g.map(sns.scatterplot, 'X', 'Y')
g.add_legend()

# Add title
plt.suptitle('Faceting with FacetGrid: Scatter Plot by Group', y=1.05)

# Save the plot
plt.savefig('faceting_facetgrid.png')

# Create a catplot for comparison
sns.catplot(x='Category', y='Y', col='Group', kind='box', data=data)

# Add title for catplot
plt.suptitle('Faceting with catplot: Box Plot by Group', y=1.05)

# Save the catplot
plt.savefig('faceting_catplot.png')