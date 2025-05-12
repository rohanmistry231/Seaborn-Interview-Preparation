import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create synthetic data
data = pd.DataFrame({
    'Category': ['A', 'B', 'C'],
    'Value': [10, 20, 15]
})

# Create a bar plot using Seaborn
sns.barplot(x='Category', y='Value', data=data, color='skyblue')

# Add labels and title
plt.title('Bar Plot (barplot): Values by Category')
plt.xlabel('Category')
plt.ylabel('Value')

# Save the plot
plt.savefig('barplot.png')