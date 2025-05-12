import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], 100),
    'Value': np.random.normal(0, 1, 100)
})

# Set a custom color palette
custom_palette = sns.color_palette("Set2", 3)
sns.set_palette(custom_palette)

# Create a box plot with the custom palette
sns.boxplot(x='Category', y='Value', data=data)

# Add labels and title
plt.title('Color Palettes: Box Plot with Set2 Palette')
plt.xlabel('Category')
plt.ylabel('Value')

# Save the plot
plt.savefig('color_palettes.png')