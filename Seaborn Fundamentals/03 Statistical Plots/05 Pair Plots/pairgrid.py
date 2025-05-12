import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'A': np.random.normal(0, 1, 50),
    'B': np.random.normal(0, 1, 50),
    'C': np.random.normal(0, 1, 50)
})

# Create a PairGrid for custom pair plots
g = sns.PairGrid(data)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot)

# Add title
plt.suptitle('PairGrid: Custom Pair Plots', y=1.02)

# Save the plot
plt.savefig('pairgrid.png')