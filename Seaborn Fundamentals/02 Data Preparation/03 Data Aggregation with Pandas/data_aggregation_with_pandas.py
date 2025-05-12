import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a synthetic DataFrame
data = pd.DataFrame({
    'Region': ['North', 'South', 'North', 'South', 'East'],
    'Sales': [200, 150, 300, 250, 180],
    'Year': [2023, 2023, 2024, 2024, 2023]
})

# Aggregate data by Region and Year
aggregated_data = data.groupby(['Region', 'Year'])['Sales'].sum().reset_index()

# Display the aggregated DataFrame
print("Aggregated Data:")
print(aggregated_data)

# Visualize the aggregated data using Seaborn
sns.barplot(x='Region', y='Sales', hue='Year', data=aggregated_data)

# Add labels and title
plt.title('Data Aggregation with Pandas: Sales by Region and Year')
plt.xlabel('Region')
plt.ylabel('Total Sales')

# Save the plot
plt.savefig('data_aggregation_with_pandas.png')