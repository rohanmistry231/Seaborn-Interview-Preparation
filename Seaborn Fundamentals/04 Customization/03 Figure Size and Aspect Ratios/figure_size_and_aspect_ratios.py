import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'X': np.random.uniform(0, 10, 50),
    'Y': np.random.uniform(0, 20, 50)
})

# Set figure size
plt.figure(figsize=(10, 4))

# Create a scatter plot with a specific aspect ratio
sns.scatterplot(x='X', y='Y', data=data, color='skyblue')

# Add labels and title
plt.title('Figure Size and Aspect Ratios: Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')

# Save the plot
plt.savefig('figure_size_and_aspect_ratios.png')