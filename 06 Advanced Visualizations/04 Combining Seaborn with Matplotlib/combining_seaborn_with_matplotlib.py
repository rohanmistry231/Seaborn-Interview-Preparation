import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set Seaborn style
sns.set_style('whitegrid')

# Create synthetic data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create a Seaborn histogram
sns.histplot(data, bins=30, color='skyblue', stat='density')

# Add a Matplotlib KDE line on top
from scipy.stats import gaussian_kde
kde = gaussian_kde(data)
x_range = np.linspace(min(data), max(data), 200)
plt.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE (Matplotlib)')

# Add labels, title, and legend
plt.title('Combining Seaborn with Matplotlib: Histogram with KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Save the plot
plt.savefig('combining_seaborn_with_matplotlib.png')