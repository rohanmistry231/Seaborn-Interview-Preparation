import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], 50),
    'Value': np.random.normal(0, 1, 50)
})

# Create a swarm plot using Seaborn
sns.swarmplot(x='Category', y='Value', data=data, color='skyblue')

# Add labels and title
plt.title('Swarm Plot (swarmplot): Values by Category')
plt.xlabel('Category')
plt.ylabel('Value')

# Save the plot
plt.savefig('swarmplot.png')