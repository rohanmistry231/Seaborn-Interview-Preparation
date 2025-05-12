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
    'Group': np.random.choice(['G1', 'G2', 'G3'], 100)
})

# Create a FacetGrid plot
g = sns.FacetGrid(data, col='Group', row='Category', margin_titles=True)
g.map(sns.scatterplot, 'X', 'Y')

# Add title
plt.suptitle('FacetGrid: Multi-Faceted Scatter Plots by Group and Category', y=1.05)

# Save the plot
plt.savefig('facetgrid_multi_faceted_plots.png')