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

# Create a pair plot using Seaborn
sns.pairplot(data)

# Add title
plt.suptitle('Pair Plot (pairplot): Pairwise Relationships', y=1.02)

# Save the plot
plt.savefig('pairplot.png')