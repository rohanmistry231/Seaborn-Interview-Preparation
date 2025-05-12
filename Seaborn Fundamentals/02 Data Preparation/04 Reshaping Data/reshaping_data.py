import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a synthetic DataFrame
data = pd.DataFrame({
    'Date': ['2023-01', '2023-01', '2023-02', '2023-02'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 120, 180]
})

# Pivot the data
pivoted_data = data.pivot(index='Date', columns='Product', values='Sales')
print("Pivoted Data:")
print(pivoted_data)

# Melt the pivoted data back to long format
melted_data = pivoted_data.reset_index().melt(id_vars='Date', var_name='Product', value_name='Sales')
print("\nMelted Data:")
print(melted_data)

# Visualize the melted data using Seaborn
sns.lineplot(x='Date', y='Sales', hue='Product', data=melted_data, marker='o')

# Add labels and title
plt.title('Reshaping Data: Sales by Product Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')

# Save the plot
plt.savefig('reshaping_data.png')