import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set Seaborn aesthetics
sns.set_style('whitegrid')  # Set background style
sns.set_context('notebook', font_scale=1.2)  # Set context for readability

# Create synthetic data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create a KDE plot with customized aesthetics
sns.kdeplot(data, color='skyblue', linewidth=2)

# Add labels and title
plt.title('Customizing Plot Aesthetics: KDE Plot')
plt.xlabel('Value')
plt.ylabel('Density')

# Save the plot
plt.savefig('customizing_plot_aesthetics.png')