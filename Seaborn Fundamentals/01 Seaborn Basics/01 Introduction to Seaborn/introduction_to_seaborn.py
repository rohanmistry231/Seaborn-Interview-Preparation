import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create a simple histogram using Seaborn
sns.histplot(data, bins=30, color='skyblue')

# Add labels and title
plt.title('Introduction to Seaborn: Histogram of Synthetic Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Save the plot
plt.savefig('introduction_to_seaborn.png')