import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'A': np.random.normal(0, 1, 50),
    'B': np.random.normal(0, 1, 50),
    'C': np.random.normal(0, 1, 50),
    'Category': np.random.choice(['X', 'Y'], 50)
})

# Create a PairGrid for pairwise relationships
g = sns.PairGrid(data, hue='Category')
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot)
g.add_legend()

# Add title
plt.suptitle('PairGrid: Pairwise Relationships with Hue', y=1.02)

# Save the plot
plt.savefig('pairgrid_pairwise_relationships.png')