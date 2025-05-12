import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create synthetic data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create a histogram using Seaborn
sns.histplot(data, bins=30, color='skyblue')

# Customize axis labels and title
plt.title('Axis Labels and Titles: Distribution of Data', fontsize=14, pad=15)
plt.xlabel('Measurement Value', fontsize=12)
plt.ylabel('Frequency Count', fontsize=12)

# Save the plot
plt.savefig('axis_labels_and_titles.png')