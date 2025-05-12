import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create synthetic time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.normal(0, 1, len(dates)).cumsum()  # Cumulative sum for trend
data = pd.DataFrame({'Date': dates, 'Value': values})

# Create a line plot for time series visualization using Seaborn
sns.lineplot(x='Date', y='Value', data=data, color='skyblue')

# Add labels and title
plt.title('Time Series Visualization: Synthetic Daily Values')
plt.xlabel('Date')
plt.ylabel('Value')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Save the plot
plt.savefig('time_series_visualization.png')