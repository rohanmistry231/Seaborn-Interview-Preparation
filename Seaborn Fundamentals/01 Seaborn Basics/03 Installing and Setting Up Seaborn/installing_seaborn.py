# Note: This file assumes Seaborn is already installed in the environment.
# In a real setup, you would install Seaborn using: pip install seaborn

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a simple dataset using Pandas
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'A', 'B', 'C'],
    'Value': [10, 20, 15, 25, 30, 22]
})

# Create a bar plot to confirm Seaborn is working
sns.barplot(x='Category', y='Value', data=data, color='skyblue')

# Add labels and title
plt.title('Installing Seaborn: Sample Bar Plot')
plt.xlabel('Category')
plt.ylabel('Value')

# Save the plot
plt.savefig('installing_seaborn.png')