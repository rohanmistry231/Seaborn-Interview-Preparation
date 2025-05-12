import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create synthetic data using NumPy arrays
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
category = np.random.choice(['A', 'B'], size=100)

# Create a DataFrame from NumPy arrays
data = pd.DataFrame({
    'X': x,
    'Y': y,
    'Category': category
})

# Display the DataFrame
print("DataFrame from NumPy Arrays:")
print(data.head())

# Visualize the data using Seaborn
sns.scatterplot(x='X', y='Y', hue='Category', data=data)

# Add labels and title
plt.title('Integration with NumPy Arrays: Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')

# Save the plot
plt.savefig('integration_with_numpy_arrays.png')