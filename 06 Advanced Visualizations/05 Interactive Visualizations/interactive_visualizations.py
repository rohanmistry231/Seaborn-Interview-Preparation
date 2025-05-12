import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px

# Create synthetic data using Pandas (as prepared with Seaborn workflow)
np.random.seed(42)
data = pd.DataFrame({
    'X': np.random.uniform(0, 10, 100),
    'Y': np.random.uniform(0, 20, 100),
    'Category': np.random.choice(['A', 'B'], 100)
})

# Create an interactive scatter plot using Plotly
# Note: Plotly generates interactive HTML outputs; here we save a static PNG
fig = px.scatter(data, x='X', y='Y', color='Category',
                 title='Interactive Visualizations: Scatter Plot (Plotly)',
                 labels={'X': 'X', 'Y': 'Y'})

# Save the plot as a static PNG
fig.write_image('interactive_visualizations.png')

# Note: In a local environment, you can use fig.show() to view the interactive plot in a browser