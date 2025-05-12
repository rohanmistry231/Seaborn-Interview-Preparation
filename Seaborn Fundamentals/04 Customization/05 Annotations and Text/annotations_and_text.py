import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'Category': ['A', 'B', 'C'],
    'Value': [10, 20, 15]
})

# Create a bar plot
ax = sns.barplot(x='Category', y='Value', data=data, color='skyblue')

# Add annotations on top of each bar
for i, v in enumerate(data['Value']):
    ax.text(i, v + 0.5, str(v), ha='center', fontsize=12)

# Add labels and title
plt.title('Annotations and Text: Bar Plot with Values')
plt.xlabel('Category')
plt.ylabel('Value')

# Save the plot
plt.savefig('annotations_and_text.png')