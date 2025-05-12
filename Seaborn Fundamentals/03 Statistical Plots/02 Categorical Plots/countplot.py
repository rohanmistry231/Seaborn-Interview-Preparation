import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create synthetic data
data = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'C', 'B', 'A']
})

# Create a count plot using Seaborn
sns.countplot(x='Category', data=data, color='skyblue')

# Add labels and title
plt.title('Count Plot (countplot): Frequency of Categories')
plt.xlabel('Category')
plt.ylabel('Count')

# Save the plot
plt.savefig('countplot.png')