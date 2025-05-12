import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a synthetic DataFrame
data = pd.DataFrame({
    'City': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Chicago'],
    'Temperature': [25, 30, 22, 27, 20],
    'Humidity': [60, 55, 65, 58, 70]
})

# Display the DataFrame
print("Original DataFrame:")
print(data)

# Use Seaborn to visualize the data
sns.scatterplot(x='Temperature', y='Humidity', hue='City', data=data)

# Add labels and title
plt.title('Working with Pandas DataFrames: Temperature vs. Humidity')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')

# Save the plot
plt.savefig('working_with_pandas_dataframes.png')