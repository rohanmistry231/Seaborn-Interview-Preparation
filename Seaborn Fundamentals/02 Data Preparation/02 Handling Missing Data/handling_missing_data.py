import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create a synthetic DataFrame with missing values
data = pd.DataFrame({
    'Age': [25, 30, np.nan, 22, 35],
    'Salary': [50000, np.nan, 60000, 45000, 70000],
    'Department': ['HR', 'IT', 'Finance', 'HR', 'IT']
})

# Display the DataFrame with missing values
print("DataFrame with Missing Values:")
print(data)

# Handle missing data by filling with the mean
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Salary'] = data['Salary'].fillna(data['Salary'].mean())

# Display the DataFrame after handling missing data
print("\nDataFrame after Handling Missing Data:")
print(data)

# Visualize the data using Seaborn
sns.boxplot(x='Department', y='Salary', data=data)

# Add labels and title
plt.title('Handling Missing Data: Salary by Department')
plt.xlabel('Department')
plt.ylabel('Salary')

# Save the plot
plt.savefig('handling_missing_data.png')