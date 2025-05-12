import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set Seaborn theme and style
sns.set_style('darkgrid')
sns.set_context('talk')  # Larger text for presentations

# Create synthetic data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create a histogram with custom styling
sns.histplot(data, bins=30, color='skyblue', stat='density')
sns.kdeplot(data, color='red', linewidth=2)

# Add labels and title
plt.title('Themes and Styling: Histogram with KDE')
plt.xlabel('Value')
plt.ylabel('Density')

# Save the plot
plt.savefig('themes_and_styling.png')