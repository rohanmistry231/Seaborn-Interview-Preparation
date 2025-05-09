# Seaborn Interview Questions for AI/ML Roles

This README provides **170 Seaborn interview questions** tailored for AI/ML students preparing for technical interviews, focusing on Seaborn’s role in creating high-quality visualizations for machine learning tasks, such as plotting model performance, feature distributions, and prediction uncertainties. The questions are categorized into **Seaborn Basics**, **Plot Customization**, **Statistical Plots**, **Categorical Plots**, **Distribution Plots**, **Matrix Plots**, **Advanced Visualizations**, and **Integration with AI/ML**. Each category is divided into **Basic**, **Intermediate**, and **Advanced** levels, with practical code snippets using Python, Seaborn, Matplotlib, NumPy, and Pandas. This resource supports candidates aiming for roles that combine data analysis and visualization with AI/ML workflows, such as data scientists or ML engineers.

## Seaborn Basics

### Basic
1. **What is Seaborn, and why is it used in AI/ML?**  
   Seaborn is a Python library for statistical data visualization, built on Matplotlib.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.lineplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.savefig('basic_lineplot.png')
   ```

2. **How do you install Seaborn?**  
   Uses pip to install.  
   ```python
   # Run in terminal: pip install seaborn
   import seaborn as sns
   print(sns.__version__)
   ```

3. **What is the role of Matplotlib in Seaborn?**  
   Seaborn uses Matplotlib as its plotting backend.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.savefig('scatterplot.png')
   ```

4. **How do you set Seaborn’s default style?**  
   Uses `sns.set_theme`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.set_theme(style="darkgrid")
   sns.lineplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.savefig('styled_lineplot.png')
   ```

5. **How do you create a simple line plot in Seaborn?**  
   Uses `sns.lineplot`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.lineplot(x=[1, 2, 3], y=[2, 4, 6])
   plt.savefig('simple_lineplot.png')
   ```

6. **How do you visualize basic plot performance?**  
   Plots rendering time (mock).  
   ```python
   import matplotlib.pyplot as plt
   plt.plot([1, 2, 3], [10, 20, 15], 'o-', label='Render Time (ms)')
   plt.title('Plot Performance')
   plt.savefig('plot_performance.png')
   ```

#### Intermediate
7. **How do you use Pandas DataFrames with Seaborn?**  
   Integrates DataFrames for plotting.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
   sns.scatterplot(data=df, x='x', y='y')
   plt.savefig('df_scatterplot.png')
   ```

8. **How do you create a scatter plot with Seaborn?**  
   Uses `sns.scatterplot`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6], size=[10, 20, 30])
   plt.savefig('sized_scatterplot.png')
   ```

9. **How do you change Seaborn’s color palette?**  
   Uses `sns.set_palette`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.set_palette("deep")
   sns.lineplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.savefig('colored_lineplot.png')
   ```

10. **How do you save Seaborn plots?**  
    Uses `plt.savefig`.  
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6])
    plt.savefig('saved_scatterplot.png')
    ```

11. **How do you create a simple bar plot?**  
    Uses `sns.barplot`.  
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.barplot(x=[1, 2, 3], y=[4, 5, 6])
    plt.savefig('barplot.png')
    ```

12. **How do you handle missing data in Seaborn?**  
    Filters NaN values with Pandas.  
    ```python
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame({'x': [1, 2, None], 'y': [4, 5, 6]}).dropna()
    sns.scatterplot(data=df, x='x', y='y')
    plt.savefig('no_missing_scatterplot.png')
    ```

#### Advanced
13. **How do you integrate Seaborn with NumPy arrays?**  
    Converts arrays to DataFrames.  
    ```python
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    data = np.array([[1, 4], [2, 5], [3, 6]])
    df = pd.DataFrame(data, columns=['x', 'y'])
    sns.lineplot(data=df, x='x', y='y')
    plt.savefig('numpy_lineplot.png')
    ```

14. **How do you use Seaborn’s context settings?**  
    Adjusts plot scale with `sns.set_context`.  
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_context("talk")
    sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6])
    plt.savefig('context_scatterplot.png')
    ```

15. **How do you create multi-panel plots?**  
    Uses `sns.FacetGrid`.  
    ```python
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame({'x': [1, 2, 3, 1, 2, 3], 'y': [4, 5, 6, 7, 8, 9], 'cat': ['A', 'A', 'A', 'B', 'B', 'B']})
    g = sns.FacetGrid(df, col="cat")
    g.map(sns.scatterplot, "x", "y")
    plt.savefig('facet_grid.png')
    ```

16. **How do you optimize Seaborn plot rendering?**  
    Reduces data points.  
    ```python
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame({'x': range(1000), 'y': range(1000)}).sample(100)
    sns.scatterplot(data=df, x='x', y='y')
    plt.savefig('optimized_scatterplot.png')
    ```

17. **How do you create custom Seaborn themes?**  
    Uses `sns.set_theme` with parameters.  
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style="white", palette="muted")
    sns.lineplot(x=[1, 2, 3], y=[4, 5, 6])
    plt.savefig('custom_theme_lineplot.png')
    ```

18. **How do you visualize Seaborn plot complexity?**  
    Plots rendering time by data size.  
    ```python
    import matplotlib.pyplot as plt
    plt.plot([100, 1000, 10000], [10, 50, 200], 'o-', label='Render Time (ms)')
    plt.title('Seaborn Plot Complexity')
    plt.savefig('plot_complexity.png')
    ```

## Plot Customization

### Basic
19. **How do you add titles to Seaborn plots?**  
   Uses `plt.title`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.title('Scatter Plot')
   plt.savefig('titled_scatterplot.png')
   ```

20. **How do you customize axis labels?**  
   Uses `plt.xlabel` and `plt.ylabel`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.lineplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.xlabel('X Axis')
   plt.ylabel('Y Axis')
   plt.savefig('labeled_lineplot.png')
   ```

21. **How do you change plot sizes?**  
   Uses `plt.figure`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   plt.figure(figsize=(8, 6))
   sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.savefig('sized_scatterplot.png')
   ```

22. **How do you add legends to Seaborn plots?**  
   Uses `hue` and `plt.legend`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': [1, 2, 3, 1, 2, 3], 'y': [4, 5, 6, 7, 8, 9], 'cat': ['A', 'A', 'A', 'B', 'B', 'B']})
   sns.scatterplot(data=df, x='x', y='y', hue='cat')
   plt.savefig('legend_scatterplot.png')
   ```

23. **How do you customize marker styles?**  
   Uses `style` parameter.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'type': ['A', 'B', 'A']})
   sns.scatterplot(data=df, x='x', y='y', style='type')
   plt.savefig('marker_scatterplot.png')
   ```

24. **How do you visualize customization impact?**  
   Plots rendering time by customization level.  
   ```python
   import matplotlib.pyplot as plt
   plt.bar(['Basic', 'Custom'], [10, 15], label='Render Time (ms)')
   plt.title('Customization Impact')
   plt.savefig('customization_impact.png')
   ```

#### Intermediate
25. **How do you customize Seaborn color scales?**  
   Uses `palette` parameter.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6], hue=[1, 2, 3], palette='viridis')
   plt.savefig('color_scale_scatterplot.png')
   ```

26. **How do you add annotations to Seaborn plots?**  
   Uses `plt.text`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.text(2, 5, 'Point')
   plt.savefig('annotated_scatterplot.png')
   ```

27. **How do you customize grid lines?**  
   Uses `sns.set_theme` with grid options.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.set_theme(style="ticks")
   sns.lineplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.grid(True)
   plt.savefig('grid_lineplot.png')
   ```

28. **How do you rotate axis labels?**  
   Uses `plt.xticks` with rotation.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.barplot(x=['A', 'B', 'C'], y=[4, 5, 6])
   plt.xticks(rotation=45)
   plt.savefig('rotated_barplot.png')
   ```

29. **How do you customize plot borders?**  
   Uses `sns.despine`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6])
   sns.despine()
   plt.savefig('borderless_scatterplot.png')
   ```

30. **How do you add error bars to Seaborn plots?**  
   Uses `sns.barplot` with error bars.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': ['A', 'B'], 'y': [4, 5], 'err': [0.5, 0.3]})
   sns.barplot(data=df, x='x', y='y', yerr=df['err'])
   plt.savefig('error_barplot.png')
   ```

#### Advanced
31. **How do you create custom color maps?**  
   Uses `sns.diverging_palette`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   cmap = sns.diverging_palette(220, 20, as_cmap=True)
   sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6], hue=[1, 2, 3], palette=cmap)
   plt.savefig('custom_cmap_scatterplot.png')
   ```

32. **How do you customize facet grid layouts?**  
   Adjusts `FacetGrid` parameters.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': [1, 2, 3, 1, 2, 3], 'y': [4, 5, 6, 7, 8, 9], 'cat': ['A', 'A', 'A', 'B', 'B', 'B']})
   g = sns.FacetGrid(df, col="cat", height=4, aspect=1.2)
   g.map(sns.scatterplot, "x", "y")
   plt.savefig('custom_facet_grid.png')
   ```

33. **How do you create interactive Seaborn plots?**  
   Uses Matplotlib with annotations (mock interactivity).  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6])
   for i, (x, y) in enumerate(zip([1, 2, 3], [4, 5, 6])):
       plt.text(x, y, f'P{i}')
   plt.savefig('interactive_scatterplot.png')
   ```

34. **How do you optimize plot customization?**  
   Minimizes rendering overhead.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6]) # Minimal customization
   plt.savefig('optimized_custom_scatterplot.png')
   ```

35. **How do you create publication-quality plots?**  
   Uses high-resolution settings.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   plt.figure(figsize=(8, 6), dpi=300)
   sns.lineplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.savefig('publication_lineplot.png')
   ```

36. **How do you visualize customization complexity?**  
   Plots rendering time by customization.  
   ```python
   import matplotlib.pyplot as plt
   plt.bar(['Simple', 'Complex'], [10, 20], label='Render Time (ms)')
   plt.title('Customization Complexity')
   plt.savefig('customization_complexity.png')
   ```

## Statistical Plots

### Basic
37. **How do you create a regression plot in Seaborn?**  
   Uses `sns.regplot`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.regplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.savefig('regplot.png')
   ```

38. **How do you create a box plot?**  
   Uses `sns.boxplot`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.boxplot(data=df, x='cat', y='val')
   plt.savefig('boxplot.png')
   ```

39. **How do you create a violin plot?**  
   Uses `sns.violinplot`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.violinplot(data=df, x='cat', y='val')
   plt.savefig('violinplot.png')
   ```

40. **How do you create a swarm plot?**  
   Uses `sns.swarmplot`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.swarmplot(data=df, x='cat', y='val')
   plt.savefig('swarmplot.png')
   ```

41. **How do you create a point plot?**  
   Uses `sns.pointplot`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.pointplot(data=df, x='cat', y='val')
   plt.savefig('pointplot.png')
   ```

42. **How do you visualize statistical plot accuracy?**  
   Plots error metrics.  
   ```python
   import matplotlib.pyplot as plt
   plt.plot([1, 2, 3], [0.1, 0.2, 0.15], 'o-', label='Error')
   plt.title('Statistical Plot Accuracy')
   plt.savefig('stat_accuracy.png')
   ```

#### Intermediate
43. **How do you create a residual plot?**  
   Uses `sns.residplot`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.residplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.savefig('residplot.png')
   ```

44. **How do you add confidence intervals to plots?**  
   Uses `sns.regplot` with `ci`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.regplot(x=[1, 2, 3, 4], y=[4, 5, 6, 5], ci=95)
   plt.savefig('ci_regplot.png')
   ```

45. **How do you create a joint plot with regression?**  
   Uses `sns.jointplot`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
   sns.jointplot(data=df, x='x', y='y', kind='reg')
   plt.savefig('joint_regplot.png')
   ```

46. **How do you create a pair plot for multiple variables?**  
   Uses `sns.pairplot`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})
   sns.pairplot(df)
   plt.savefig('pairplot.png')
   ```

47. **How do you customize statistical annotations?**  
   Uses `sns.regplot` with annotations.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.regplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.text(2, 5, 'Fit Line')
   plt.savefig('annotated_regplot.png')
   ```

48. **How do you create a strip plot?**  
   Uses `sns.stripplot`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.stripplot(data=df, x='cat', y='val')
   plt.savefig('stripplot.png')
   ```

#### Advanced
49. **How do you create a complex regression plot?**  
   Uses `sns.lmplot` with facets.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': [1, 2, 3, 1, 2, 3], 'y': [4, 5, 6, 7, 8, 9], 'cat': ['A', 'A', 'A', 'B', 'B', 'B']})
   sns.lmplot(data=df, x='x', y='y', col='cat')
   plt.savefig('complex_lmplot.png')
   ```

50. **How do you visualize model residuals?**  
   Uses `sns.residplot` with customization.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.residplot(x=[1, 2, 3, 4], y=[4, 5, 6, 5], lowess=True)
   plt.savefig('custom_residplot.png')
   ```

51. **How do you create a partial regression plot?**  
   Uses `sns.regplot` with partial data (mock).  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.regplot(x=[1, 2, 3], y=[4, 5, 6])
   plt.savefig('partial_regplot.png')
   ```

52. **How do you optimize statistical plot rendering?**  
   Reduces data points.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': range(1000), 'y': range(1000)}).sample(100)
   sns.regplot(data=df, x='x', y='y')
   plt.savefig('optimized_regplot.png')
   ```

53. **How do you create a robust regression plot?**  
   Uses `sns.regplot` with robust fitting.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.regplot(x=[1, 2, 3, 10], y=[4, 5, 6, 20], robust=True)
   plt.savefig('robust_regplot.png')
   ```

54. **How do you visualize statistical plot performance?**  
   Plots regression fit time.  
   ```python
   import matplotlib.pyplot as plt
   plt.plot([100, 1000], [50, 200], 'o-', label='Fit Time (ms)')
   plt.title('Statistical Plot Performance')
   plt.savefig('stat_performance.png')
   ```

## Categorical Plots

### Basic
55. **How do you create a categorical scatter plot?**  
   Uses `sns.catplot` with `kind='scatter'`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.catplot(data=df, x='cat', y='val', kind='scatter')
   plt.savefig('cat_scatterplot.png')
   ```

56. **How do you create a count plot?**  
   Uses `sns.countplot`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B', 'A']})
   sns.countplot(data=df, x='cat')
   plt.savefig('countplot.png')
   ```

57. **How do you create a categorical bar plot?**  
   Uses `sns.catplot` with `kind='bar'`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.catplot(data=df, x='cat', y='val', kind='bar')
   plt.savefig('cat_barplot.png')
   ```

58. **How do you create a categorical box plot?**  
   Uses `sns.catplot` with `kind='box'`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.catplot(data=df, x='cat', y='val', kind='box')
   plt.savefig('cat_boxplot.png')
   ```

59. **How do you create a categorical violin plot?**  
   Uses `sns.catplot` with `kind='violin'`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.catplot(data=df, x='cat', y='val', kind='violin')
   plt.savefig('cat_violinplot.png')
   ```

60. **How do you visualize categorical plot clarity?**  
   Plots data density.  
   ```python
   import matplotlib.pyplot as plt
   plt.bar(['A', 'B'], [10, 15], label='Data Points')
   plt.title('Categorical Plot Clarity')
   plt.savefig('cat_clarity.png')
   ```

#### Intermediate
61. **How do you create a categorical point plot?**  
   Uses `sns.catplot` with `kind='point'`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.catplot(data=df, x='cat', y='val', kind='point')
   plt.savefig('cat_pointplot.png')
   ```

62. **How do you create a categorical swarm plot?**  
   Uses `sns.catplot` with `kind='swarm'`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.catplot(data=df, x='cat', y='val', kind='swarm')
   plt.savefig('cat_swarmplot.png')
   ```

63. **How do you add hue to categorical plots?**  
   Uses `hue` parameter.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4], 'type': ['X', 'Y', 'X', 'Y']})
   sns.catplot(data=df, x='cat', y='val', hue='type', kind='bar')
   plt.savefig('hue_cat_barplot.png')
   ```

64. **How do you create faceted categorical plots?**  
   Uses `sns.catplot` with `col`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4], 'group': ['X', 'X', 'Y', 'Y']})
   sns.catplot(data=df, x='cat', y='val', col='group', kind='bar')
   plt.savefig('faceted_cat_barplot.png')
   ```

65. **How do you customize categorical plot aesthetics?**  
   Uses `palette` and `style`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4], 'type': ['X', 'Y', 'X', 'Y']})
   sns.catplot(data=df, x='cat', y='val', hue='type', kind='point', palette='muted')
   plt.savefig('custom_cat_pointplot.png')
   ```

66. **How do you handle large categorical datasets?**  
   Aggregates data.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A']*100 + ['B']*100, 'val': list(range(100)) + list(range(100))})
   sns.catplot(data=df.groupby('cat').mean().reset_index(), x='cat', y='val', kind='bar')
   plt.savefig('large_cat_barplot.png')
   ```

#### Advanced
67. **How do you create complex categorical plots?**  
   Combines `hue`, `col`, and `row`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({
       'cat': ['A', 'A', 'B', 'B']*2,
       'val': [1, 2, 3, 4]*2,
       'type': ['X', 'Y', 'X', 'Y']*2,
       'group': ['G1', 'G1', 'G1', 'G1', 'G2', 'G2', 'G2', 'G2']
   })
   sns.catplot(data=df, x='cat', y='val', hue='type', col='group', kind='bar')
   plt.savefig('complex_cat_barplot.png')
   ```

68. **How do you create stacked bar plots?**  
   Uses Pandas with `sns.catplot` (mock).  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'B'], 'val1': [1, 3], 'val2': [2, 4]})
   df_melt = df.melt(id_vars='cat', value_vars=['val1', 'val2'], var_name='type')
   sns.catplot(data=df_melt, x='cat', y='value', hue='type', kind='bar')
   plt.savefig('stacked_barplot.png')
   ```

69. **How do you optimize categorical plot rendering?**  
   Reduces categories.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.catplot(data=df, x='cat', y='val', kind='bar')
   plt.savefig('optimized_cat_barplot.png')
   ```

70. **How do you create categorical plots with error bars?**  
   Uses `sns.catplot` with `yerr`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'cat': ['A', 'B'], 'val': [4, 5], 'err': [0.5, 0.3]})
   sns.barplot(data=df, x='cat', y='val', yerr=df['err'])
   plt.savefig('cat_error_barplot.png')
   ```

71. **How do you create publication-quality categorical plots?**  
   Uses high-resolution settings.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   plt.figure(figsize=(8, 6), dpi=300)
   df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
   sns.catplot(data=df, x='cat', y='val', kind='box')
   plt.savefig('pub_cat_boxplot.png')
   ```

72. **How do you visualize categorical plot performance?**  
   Plots rendering time by category count.  
   ```python
   import matplotlib.pyplot as plt
   plt.plot([2, 10, 50], [10, 20, 50], 'o-', label='Render Time (ms)')
   plt.title('Categorical Plot Performance')
   plt.savefig('cat_performance.png')
   ```

## Distribution Plots

### Basic
73. **How do you create a histogram in Seaborn?**  
   Uses `sns.histplot`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.histplot(data=[1, 2, 2, 3, 3, 3])
   plt.savefig('histogram.png')
   ```

74. **How do you create a kernel density plot?**  
   Uses `sns.kdeplot`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.kdeplot(data=[1, 2, 2, 3, 3, 3])
   plt.savefig('kdeplot.png')
   ```

75. **How do you create a rug plot?**  
   Uses `sns.rugplot`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.rugplot(data=[1, 2, 2, 3, 3, 3])
   plt.savefig('rugplot.png')
   ```

76. **How do you create a distribution plot?**  
   Uses `sns.distplot` (deprecated, use `histplot`).  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.histplot(data=[1, 2, 2, 3, 3, 3], kde=True)
   plt.savefig('distplot.png')
   ```

77. **How do you create an ECDF plot?**  
   Uses `sns.ecdfplot`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.ecdfplot(data=[1, 2, 2, 3, 3, 3])
   plt.savefig('ecdfplot.png')
   ```

78. **How do you visualize distribution plot accuracy?**  
   Plots binning errors.  
   ```python
   import matplotlib.pyplot as plt
   plt.plot([10, 50, 100], [0.1, 0.05, 0.02], 'o-', label='Bin Error')
   plt.title('Distribution Plot Accuracy')
   plt.savefig('dist_accuracy.png')
   ```

#### Intermediate
79. **How do you create a bivariate KDE plot?**  
   Uses `sns.kdeplot` with two variables.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
   sns.kdeplot(data=df, x='x', y='y')
   plt.savefig('bivariate_kdeplot.png')
   ```

80. **How do you customize histogram bins?**  
   Uses `bins` parameter.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.histplot(data=[1, 2, 2, 3, 3, 3], bins=5)
   plt.savefig('custom_histogram.png')
   ```

81. **How do you create a faceted distribution plot?**  
   Uses `sns.FacetGrid` with `histplot`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'val': [1, 2, 3, 1, 2, 3], 'cat': ['A', 'A', 'A', 'B', 'B', 'B']})
   g = sns.FacetGrid(df, col="cat")
   g.map(sns.histplot, "val")
   plt.savefig('faceted_histogram.png')
   ```

82. **How do you combine distribution plots?**  
   Uses `sns.histplot` with `kde`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.histplot(data=[1, 2, 2, 3, 3, 3], kde=True)
   plt.savefig('combined_distplot.png')
   ```

83. **How do you create a cumulative histogram?**  
   Uses `sns.histplot` with `cumulative`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.histplot(data=[1, 2, 2, 3, 3, 3], cumulative=True)
   plt.savefig('cumulative_histogram.png')
   ```

84. **How do you handle large distribution datasets?**  
   Uses sampling.  
   ```python
   import seaborn as sns
   import numpy as np
   import matplotlib.pyplot as plt
   data = np.random.randn(10000)[:1000]
   sns.histplot(data=data)
   plt.savefig('large_histogram.png')
   ```

#### Advanced
85. **How do you create a conditional KDE plot?**  
   Uses `sns.kdeplot` with `hue`.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'val': [1, 2, 3, 4, 5, 6], 'cat': ['A', 'A', 'A', 'B', 'B', 'B']})
   sns.kdeplot(data=df, x='val', hue='cat')
   plt.savefig('conditional_kdeplot.png')
   ```

86. **How do you create a weighted distribution plot?**  
   Uses `weights` in `sns.histplot`.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.histplot(data=[1, 2, 2, 3], weights=[1, 2, 2, 1])
   plt.savefig('weighted_histogram.png')
   ```

87. **How do you optimize distribution plot rendering?**  
   Reduces bin count.  
   ```python
   import seaborn as sns
   import numpy as np
   import matplotlib.pyplot as plt
   sns.histplot(data=np.random.randn(1000), bins=20)
   plt.savefig('optimized_histogram.png')
   ```

88. **How do you create a 3D KDE plot?**  
   Uses `sns.kdeplot` with contours (mock 3D).  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
   sns.kdeplot(data=df, x='x', y='y', fill=True)
   plt.savefig('3d_kdeplot.png')
   ```

89. **How do you create publication-quality distribution plots?**  
   Uses high-resolution settings.  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   plt.figure(figsize=(8, 6), dpi=300)
   sns.histplot(data=[1, 2, 2, 3, 3, 3], kde=True)
   plt.savefig('pub_histogram.png')
   ```

90. **How do you visualize distribution plot performance?**  
   Plots rendering time by data size.  
   ```python
   import matplotlib.pyplot as plt
   plt.plot([1000, 10000], [50, 200], 'o-', label='Render Time (ms)')
   plt.title('Distribution Plot Performance')
   plt.savefig('dist_performance.png')
   ```

## Matrix Plots

### Basic
91. **How do you create a heatmap in Seaborn?**  
   Uses `sns.heatmap`.  
   ```python
   import seaborn as sns
   import numpy as np
   import matplotlib.pyplot as plt
   data = np.array([[1, 2], [3, 4]])
   sns.heatmap(data)
   plt.savefig('heatmap.png')
   ```

92. **How do you create a correlation matrix plot?**  
   Uses `sns.heatmap` with Pandas.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
   sns.heatmap(df.corr())
   plt.savefig('corr_heatmap.png')
   ```

93. **How do you create a clustermap?**  
   Uses `sns.clustermap`.  
   ```python
   import seaborn as sns
   import numpy as np
   import matplotlib.pyplot as plt
   data = np.array([[1, 2], [3, 4]])
   sns.clustermap(data)
   plt.savefig('clustermap.png')
   ```

94. **How do you annotate heatmap cells?**  
   Uses `annot` in `sns.heatmap`.  
   ```python
   import seaborn as sns
   import numpy as np
   import matplotlib.pyplot as plt
   data = np.array([[1, 2], [3, 4]])
   sns.heatmap(data, annot=True)
   plt.savefig('annotated_heatmap.png')
   ```

95. **How do you customize heatmap colors?**  
   Uses `cmap` in `sns.heatmap`.  
   ```python
   import seaborn as sns
   import numpy as np
   import matplotlib.pyplot as plt
   data = np.array([[1, 2], [3, 4]])
   sns.heatmap(data, cmap='Blues')
   plt.savefig('colored_heatmap.png')
   ```

96. **How do you visualize matrix plot clarity?**  
   Plots cell density.  
   ```python
   import matplotlib.pyplot as plt
   plt.bar(['2x2', '10x10'], [10, 50], label='Cell Count')
   plt.title('Matrix Plot Clarity')
   plt.savefig('matrix_clarity.png')
   ```

#### Intermediate
97. **How do you create a masked heatmap?**  
   Uses `mask` in `sns.heatmap`.  
   ```python
   import seaborn as sns
   import numpy as np
   import matplotlib.pyplot as plt
   data = np.array([[1, 2], [3, 4]])
   mask = np.triu(data)
   sns.heatmap(data, mask=mask)
   plt.savefig('masked_heatmap.png')
   ```

98. **How do you create a hierarchical clustermap?**  
   Uses `sns.clustermap` with clustering.  
   ```python
   import seaborn as sns
   import pandas as pd
   import matplotlib.pyplot as plt
   df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
   sns.clustermap(df.corr())
   plt.savefig('hierarchical_clustermap.png')
   ```

99. **How do you customize heatmap scales?**  
   Uses `vmin` and `vmax`.  
   ```python
   import seaborn as sns
   import numpy as np
   import matplotlib.pyplot as plt
   data = np.array([[1, 2], [3, 4]])
   sns.heatmap(data, vmin=0, vmax=5)
   plt.savefig('scaled_heatmap.png')
   ```

100. **How do you create a heatmap with labels?**  
    Uses `xticklabels` and `yticklabels`.  
    ```python
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    data = np.array([[1, 2], [3, 4]])
    sns.heatmap(data, xticklabels=['A', 'B'], yticklabels=['X', 'Y'])
    plt.savefig('labeled_heatmap.png')
    ```

101. **How do you handle large matrix datasets?**  
     Aggregates data.  
     ```python
     import seaborn as sns
     import numpy as np
     import matplotlib.pyplot as plt
     data = np.random.rand(100, 100)[:10, :10]
     sns.heatmap(data)
     plt.savefig('large_heatmap.png')
     ```

102. **How do you create a diverging heatmap?**  
     Uses `center` in `sns.heatmap`.  
     ```python
     import seaborn as sns
     import numpy as np
     import matplotlib.pyplot as plt
     data = np.array([[1, -2], [3, -4]])
     sns.heatmap(data, center=0, cmap='RdBu')
     plt.savefig('diverging_heatmap.png')
     ```

#### Advanced
103. **How do you create a complex clustermap?**  
     Uses `sns.clustermap` with customization.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})
     sns.clustermap(df.corr(), cmap='coolwarm', annot=True)
     plt.savefig('complex_clustermap.png')
     ```

104. **How do you optimize matrix plot rendering?**  
     Reduces matrix size.  
     ```python
     import seaborn as sns
     import numpy as np
     import matplotlib.pyplot as plt
     data = np.random.rand(50, 50)[:10, :10]
     sns.heatmap(data)
     plt.savefig('optimized_heatmap.png')
     ```

105. **How do you create a publication-quality heatmap?**  
     Uses high-resolution settings.  
     ```python
     import seaborn as sns
     import numpy as np
     import matplotlib.pyplot as plt
     plt.figure(figsize=(8, 6), dpi=300)
     data = np.array([[1, 2], [3, 4]])
     sns.heatmap(data, annot=True, cmap='viridis')
     plt.savefig('pub_heatmap.png')
     ```

106. **How do you create a triangular heatmap?**  
     Uses `mask` for upper triangle.  
     ```python
     import seaborn as sns
     import numpy as np
     import matplotlib.pyplot as plt
     data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
     mask = np.triu(data)
     sns.heatmap(data, mask=mask, annot=True)
     plt.savefig('triangular_heatmap.png')
     ```

107. **How do you visualize matrix plot performance?**  
     Plots rendering time by matrix size.  
     ```python
     import matplotlib.pyplot as plt
     plt.plot([10, 50, 100], [20, 100, 500], 'o-', label='Render Time (ms)')
     plt.title('Matrix Plot Performance')
     plt.savefig('matrix_performance.png')
     ```

108. **How do you create a heatmap with clustering?**  
     Uses `sns.clustermap` with clustering options.  
     ```python
     import seaborn as sns
     import numpy as np
     import matplotlib.pyplot as plt
     data = np.array([[1, 2], [3, 4]])
     sns.clustermap(data, method='average')
     plt.savefig('clustered_heatmap.png')
     ```

## Advanced Visualizations

### Basic
109. **How do you create a joint grid plot?**  
     Uses `sns.JointGrid`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
     g = sns.JointGrid(data=df, x='x', y='y')
     g.plot(sns.scatterplot, sns.histplot)
     plt.savefig('joint_grid.png')
     ```

110. **How do you create a pair grid plot?**  
     Uses `sns.PairGrid`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})
     g = sns.PairGrid(df)
     g.map(sns.scatterplot)
     plt.savefig('pair_grid.png')
     ```

111. **How do you create a facet grid plot?**  
     Uses `sns.FacetGrid`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3, 1, 2, 3], 'y': [4, 5, 6, 7, 8, 9], 'cat': ['A', 'A', 'A', 'B', 'B', 'B']})
     g = sns.FacetGrid(df, col='cat')
     g.map(sns.scatterplot, 'x', 'y')
     plt.savefig('facet_grid_adv.png')
     ```

112. **How do you create a multi-plot figure?**  
     Uses `plt.subplots`.  
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
     sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6], ax=ax1)
     sns.lineplot(x=[1, 2, 3], y=[4, 5, 6], ax=ax2)
     plt.savefig('multi_plot.png')
     ```

113. **How do you create a 3D-like plot?**  
     Uses `sns.kdeplot` with shading (mock 3D).  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
     sns.kdeplot(data=df, x='x', y='y', fill=True)
     plt.savefig('3d_like_plot.png')
     ```

114. **How do you visualize advanced plot complexity?**  
     Plots rendering time by plot type.  
     ```python
     import matplotlib.pyplot as plt
     plt.bar(['Joint', 'Pair'], [50, 100], label='Render Time (ms)')
     plt.title('Advanced Plot Complexity')
     plt.savefig('adv_complexity.png')
     ```

#### Intermediate
115. **How do you customize joint grid plots?**  
     Uses `JointGrid` with custom plots.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
     g = sns.JointGrid(data=df, x='x', y='y')
     g.plot_joint(sns.scatterplot)
     g.plot_marginals(sns.kdeplot)
     plt.savefig('custom_joint_grid.png')
     ```

116. **How do you create a conditional pair grid?**  
     Uses `PairGrid` with `hue`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3, 1, 2, 3], 'y': [4, 5, 6, 7, 8, 9], 'cat': ['A', 'A', 'A', 'B', 'B', 'B']})
     g = sns.PairGrid(df, hue='cat')
     g.map(sns.scatterplot)
     plt.savefig('conditional_pair_grid.png')
     ```

117. **How do you create a faceted regression plot?**  
     Uses `sns.lmplot` with facets.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3, 1, 2, 3], 'y': [4, 5, 6, 7, 8, 9], 'cat': ['A', 'A', 'A', 'B', 'B', 'B']})
     sns.lmplot(data=df, x='x', y='y', col='cat')
     plt.savefig('faceted_lmplot.png')
     ```

118. **How do you combine multiple plot types?**  
     Uses `JointGrid` with mixed plots.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
     g = sns.JointGrid(data=df, x='x', y='y')
     g.plot_joint(sns.scatterplot)
     g.plot_marginals(sns.histplot)
     plt.savefig('mixed_joint_grid.png')
     ```

119. **How do you create a dynamic visualization?**  
     Uses annotations for interactivity (mock).  
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6])
     plt.text(2, 5, 'Dynamic Point')
     plt.savefig('dynamic_scatterplot.png')
     ```

120. **How do you optimize advanced visualizations?**  
     Reduces data points.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': range(1000), 'y': range(1000)}).sample(100)
     g = sns.JointGrid(data=df, x='x', y='y')
     g.plot(sns.scatterplot, sns.histplot)
     plt.savefig('optimized_joint_grid.png')
     ```

#### Advanced
121. **How do you create a complex pair grid?**  
     Uses `PairGrid` with multiple plot types.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})
     g = sns.PairGrid(df)
     g.map_diag(sns.histplot)
     g.map_offdiag(sns.scatterplot)
     plt.savefig('complex_pair_grid.png')
     ```

122. **How do you create a publication-quality joint grid?**  
     Uses high-resolution settings.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     plt.figure(figsize=(8, 6), dpi=300)
     df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
     g = sns.JointGrid(data=df, x='x', y='y')
     g.plot(sns.scatterplot, sns.histplot)
     plt.savefig('pub_joint_grid.png')
     ```

123. **How do you create a multi-faceted advanced plot?**  
     Uses `FacetGrid` with complex mappings.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'x': [1, 2, 3, 1, 2, 3],
         'y': [4, 5, 6, 7, 8, 9],
         'cat': ['A', 'A', 'A', 'B', 'B', 'B'],
         'group': ['X', 'X', 'X', 'Y', 'Y', 'Y']
     })
     g = sns.FacetGrid(df, col='cat', row='group')
     g.map(sns.scatterplot, 'x', 'y')
     plt.savefig('multi_facet_grid.png')
     ```

124. **How do you create a 3D-like advanced visualization?**  
     Uses `sns.kdeplot` with contours.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
     sns.kdeplot(data=df, x='x', y='y', fill=True, cmap='viridis')
     plt.savefig('3d_adv_kdeplot.png')
     ```

125. **How do you optimize complex visualizations?**  
     Reduces plot elements.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': range(1000), 'y': range(1000)}).sample(100)
     sns.pairplot(df)
     plt.savefig('optimized_pairplot.png')
     ```

126. **How do you visualize advanced visualization performance?**  
     Plots rendering time by plot complexity.  
     ```python
     import matplotlib.pyplot as plt
     plt.plot(['Joint', 'Pair', 'Facet'], [50, 100, 150], 'o-', label='Render Time (ms)')
     plt.title('Advanced Visualization Performance')
     plt.savefig('adv_performance.png')
     ```

## Integration with AI/ML

### Basic
127. **How do you visualize model performance with Seaborn?**  
     Plots accuracy over epochs.  
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     sns.lineplot(x=[1, 2, 3], y=[0.8, 0.85, 0.9])
     plt.title('Model Accuracy')
     plt.savefig('model_accuracy.png')
     ```

128. **How do you plot feature distributions?**  
     Uses `sns.histplot`.  
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     sns.histplot(data=[1, 2, 2, 3, 3, 3])
     plt.title('Feature Distribution')
     plt.savefig('feature_histogram.png')
     ```

129. **How do you visualize model predictions?**  
     Uses `sns.scatterplot`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'true': [1, 2, 3], 'pred': [1.1, 2.2, 2.9]})
     sns.scatterplot(data=df, x='true', y='pred')
     plt.title('Predictions vs True')
     plt.savefig('prediction_scatterplot.png')
     ```

130. **How do you plot confusion matrices?**  
     Uses `sns.heatmap`.  
     ```python
     import seaborn as sns
     import numpy as np
     import matplotlib.pyplot as plt
     cm = np.array([[10, 2], [3, 15]])
     sns.heatmap(cm, annot=True)
     plt.title('Confusion Matrix')
     plt.savefig('confusion_matrix.png')
     ```

131. **How do you visualize loss curves?**  
     Uses `sns.lineplot`.  
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     sns.lineplot(x=[1, 2, 3], y=[0.5, 0.3, 0.1])
     plt.title('Loss Curve')
     plt.savefig('loss_curve.png')
     ```

132. **How do you visualize model metrics?**  
     Plots multiple metrics.  
     ```python
     import matplotlib.pyplot as plt
     plt.plot([1, 2, 3], [0.8, 0.85, 0.9], label='Accuracy')
     plt.plot([1, 2, 3], [0.5, 0.3, 0.1], label='Loss')
     plt.legend()
     plt.title('Model Metrics')
     plt.savefig('model_metrics.png')
     ```

#### Intermediate
133. **How do you plot feature importance?**  
     Uses `sns.barplot`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'feature': ['A', 'B', 'C'], 'importance': [0.4, 0.3, 0.2]})
     sns.barplot(data=df, x='feature', y='importance')
     plt.title('Feature Importance')
     plt.savefig('feature_importance.png')
     ```

134. **How do you visualize model residuals?**  
     Uses `sns.residplot`.  
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     sns.residplot(x=[1, 2, 3, 4], y=[4, 5, 6, 5])
     plt.title('Model Residuals')
     plt.savefig('model_residuals.png')
     ```

135. **How do you plot ROC curves?**  
     Uses `sns.lineplot` (mock).  
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     sns.lineplot(x=[0, 0.5, 1], y=[0, 0.7, 1])
     plt.title('ROC Curve')
     plt.savefig('roc_curve.png')
     ```

136. **How do you visualize model uncertainty?**  
     Uses `sns.kdeplot`.  
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     sns.kdeplot(data=[0.1, 0.2, 0.2, 0.3])
     plt.title('Prediction Uncertainty')
     plt.savefig('uncertainty_kdeplot.png')
     ```

137. **How do you plot training vs validation metrics?**  
     Uses `sns.lineplot`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'epoch': [1, 2, 3, 1, 2, 3],
         'metric': [0.8, 0.85, 0.9, 0.75, 0.8, 0.85],
         'type': ['train', 'train', 'train', 'val', 'val', 'val']
     })
     sns.lineplot(data=df, x='epoch', y='metric', hue='type')
     plt.title('Train vs Val Metrics')
     plt.savefig('train_val_metrics.png')
     ```

138. **How do you visualize feature correlations?**  
     Uses `sns.heatmap`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})
     sns.heatmap(df.corr(), annot=True)
     plt.title('Feature Correlations')
     plt.savefig('feature_correlations.png')
     ```

#### Advanced
139. **How do you create a complex model performance plot?**  
     Uses `sns.FacetGrid` with metrics.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'epoch': [1, 2, 3, 1, 2, 3],
         'metric': [0.8, 0.85, 0.9, 0.75, 0.8, 0.85],
         'type': ['train', 'train', 'train', 'val', 'val', 'val'],
         'model': ['A', 'A', 'A', 'B', 'B', 'B']
     })
     g = sns.FacetGrid(df, col='model')
     g.map(sns.lineplot, 'epoch', 'metric', 'type')
     plt.savefig('complex_model_performance.png')
     ```

140. **How do you visualize model prediction distributions?**  
     Uses `sns.histplot` with `hue`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'pred': [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
         'class': ['A', 'A', 'A', 'B', 'B', 'B']
     })
     sns.histplot(data=df, x='pred', hue='class')
     plt.title('Prediction Distributions')
     plt.savefig('pred_distributions.png')
     ```

141. **How do you create a publication-quality model plot?**  
     Uses high-resolution settings.  
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     plt.figure(figsize=(8, 6), dpi=300)
     sns.lineplot(x=[1, 2, 3], y=[0.8, 0.85, 0.9])
     plt.title('Model Performance')
     plt.savefig('pub_model_performance.png')
     ```

142. **How do you visualize model comparison?**  
     Uses `sns.catplot`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'model': ['A', 'A', 'B', 'B'],
         'metric': [0.8, 0.85, 0.9, 0.88],
         'type': ['train', 'val', 'train', 'val']
     })
     sns.catplot(data=df, x='model', y='metric', hue='type', kind='bar')
     plt.title('Model Comparison')
     plt.savefig('model_comparison.png')
     ```

143. **How do you optimize ML visualization rendering?**  
     Reduces data points.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': range(1000), 'y': range(1000)}).sample(100)
     sns.scatterplot(data=df, x='x', y='y')
     plt.title('Optimized ML Plot')
     plt.savefig('optimized_ml_scatterplot.png')
     ```

144. **How do you visualize model training progress?**  
     Uses `sns.lineplot` with facets.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'epoch': [1, 2, 3, 1, 2, 3],
         'metric': [0.8, 0.85, 0.9, 0.75, 0.8, 0.85],
         'type': ['train', 'train', 'train', 'val', 'val', 'val']
     })
     sns.lineplot(data=df, x='epoch', y='metric', hue='type')
     plt.title('Training Progress')
     plt.savefig('training_progress.png')
     ```

145. **How do you create a complex confusion matrix?**  
     Uses `sns.heatmap` with customization.  
     ```python
     import seaborn as sns
     import numpy as np
     import matplotlib.pyplot as plt
     cm = np.array([[10, 2, 1], [3, 15, 0], [1, 0, 20]])
     sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=['A', 'B', 'C'], yticklabels=['A', 'B', 'C'])
     plt.title('Complex Confusion Matrix')
     plt.savefig('complex_confusion_matrix.png')
     ```

146. **How do you visualize feature interactions?**  
     Uses `sns.pairplot`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})
     sns.pairplot(df)
     plt.savefig('feature_interactions.png')
     ```

147. **How do you create a model ensemble visualization?**  
     Uses `sns.catplot` for comparison.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'model': ['A', 'B', 'Ensemble'],
         'accuracy': [0.8, 0.85, 0.9]
     })
     sns.catplot(data=df, x='model', y='accuracy', kind='bar')
     plt.title('Model Ensemble Performance')
     plt.savefig('ensemble_performance.png')
     ```

148. **How do you visualize model uncertainty distributions?**  
     Uses `sns.kdeplot` with `hue`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'uncertainty': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
         'class': ['A', 'A', 'A', 'B', 'B', 'B']
     })
     sns.kdeplot(data=df, x='uncertainty', hue='class')
     plt.title('Uncertainty Distributions')
     plt.savefig('uncertainty_distributions.png')
     ```

149. **How do you create a feature selection visualization?**  
     Uses `sns.barplot`.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({'feature': ['A', 'B', 'C'], 'score': [0.5, 0.3, 0.1]})
     sns.barplot(data=df, x='feature', y='score')
     plt.title('Feature Selection Scores')
     plt.savefig('feature_selection.png')
     ```

150. **How do you visualize model performance across datasets?**  
     Uses `sns.catplot` with facets.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'dataset': ['D1', 'D1', 'D2', 'D2'],
         'metric': [0.8, 0.85, 0.9, 0.88],
         'type': ['train', 'val', 'train', 'val']
     })
     sns.catplot(data=df, x='type', y='metric', col='dataset', kind='bar')
     plt.savefig('dataset_performance.png')
     ```

151. **How do you create a model calibration plot?**  
     Uses `sns.lineplot` to plot expected vs. observed probabilities (mock).  
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     sns.lineplot(x=[0, 0.5, 1], y=[0.1, 0.6, 0.9])
     plt.title('Model Calibration')
     plt.xlabel('Expected Probability')
     plt.ylabel('Observed Probability')
     plt.savefig('calibration_plot.png')
     ```

152. **How do you visualize model training dynamics?**  
     Uses `sns.lineplot` with multiple metrics to show training progress.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'epoch': [1, 2, 3, 1, 2, 3],
         'value': [0.5, 0.3, 0.1, 0.8, 0.85, 0.9],
         'metric': ['loss', 'loss', 'loss', 'accuracy', 'accuracy', 'accuracy']
     })
     sns.lineplot(data=df, x='epoch', y='value', hue='metric')
     plt.title('Training Dynamics')
     plt.savefig('training_dynamics.png')
     ```

153. **How do you visualize model decision boundaries?**  
     Uses `sns.scatterplot` with a contour plot (mock decision boundary).  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'x': [1, 2, 3, 4, 5],
         'y': [1, 2, 3, 2, 1],
         'class': ['A', 'A', 'B', 'B', 'A']
     })
     sns.scatterplot(data=df, x='x', y='y', hue='class')
     plt.contour([[1, 2], [3, 4]], levels=[2], colors='black')
     plt.title('Decision Boundary')
     plt.savefig('decision_boundary.png')
     ```

154. **How do you create a feature contribution plot?**  
     Uses `sns.barplot` to show feature contributions to predictions.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'feature': ['F1', 'F2', 'F3'],
         'contribution': [0.5, -0.2, 0.3]
     })
     sns.barplot(data=df, x='feature', y='contribution')
     plt.title('Feature Contributions')
     plt.savefig('feature_contribution.png')
     ```

155. **How do you visualize model performance over time?**  
     Uses `sns.lineplot` to plot metrics over time steps.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'time': [1, 2, 3, 4],
         'metric': [0.7, 0.75, 0.8, 0.85],
         'type': ['test', 'test', 'test', 'test']
     })
     sns.lineplot(data=df, x='time', y='metric', hue='type')
     plt.title('Model Performance Over Time')
     plt.savefig('performance_over_time.png')
     ```

156. **How do you create a model comparison heatmap?**  
     Uses `sns.heatmap` to compare model metrics across datasets.  
     ```python
     import seaborn as sns
     import numpy as np
     import matplotlib.pyplot as plt
     data = np.array([[0.8, 0.85], [0.9, 0.88]])
     sns.heatmap(data, annot=True, xticklabels=['Model A', 'Model B'], yticklabels=['D1', 'D2'])
     plt.title('Model Comparison Heatmap')
     plt.savefig('model_comparison_heatmap.png')
     ```

157. **How do you visualize model robustness?**  
     Uses `sns.boxplot` to show metric variability under noise.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'noise': ['Low', 'Low', 'High', 'High'],
         'metric': [0.85, 0.83, 0.75, 0.70]
     })
     sns.boxplot(data=df, x='noise', y='metric')
     plt.title('Model Robustness')
     plt.savefig('model_robustness.png')
     ```

158. **How do you create a feature interaction heatmap?**  
     Uses `sns.heatmap` to show pairwise feature interactions.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'F1': [1, 2, 3],
         'F2': [4, 5, 6],
         'F3': [7, 8, 9]
     })
     sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
     plt.title('Feature Interaction Heatmap')
     plt.savefig('feature_interaction_heatmap.png')
     ```

159. **How do you visualize model fairness metrics?**  
     Uses `sns.barplot` to compare fairness metrics across groups.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'group': ['G1', 'G2', 'G3'],
         'fairness': [0.9, 0.85, 0.88]
     })
     sns.barplot(data=df, x='group', y='fairness')
     plt.title('Model Fairness Metrics')
     plt.savefig('fairness_metrics.png')
     ```

160. **How do you create a model uncertainty heatmap?**  
     Uses `sns.heatmap` to show uncertainty across predictions.  
     ```python
     import seaborn as sns
     import numpy as np
     import matplotlib.pyplot as plt
     data = np.array([[0.1, 0.2], [0.3, 0.15]])
     sns.heatmap(data, annot=True, cmap='YlOrRd')
     plt.title('Model Uncertainty Heatmap')
     plt.savefig('uncertainty_heatmap.png')
     ```

161. **How do you visualize model performance by subgroup?**  
     Uses `sns.catplot` to compare metrics across subgroups.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'subgroup': ['A', 'A', 'B', 'B'],
         'metric': [0.8, 0.85, 0.9, 0.87],
         'type': ['train', 'val', 'train', 'val']
     })
     sns.catplot(data=df, x='subgroup', y='metric', hue='type', kind='bar')
     plt.title('Performance by Subgroup')
     plt.savefig('subgroup_performance.png')
     ```

162. **How do you create a model sensitivity plot?**  
     Uses `sns.lineplot` to show metric sensitivity to hyperparameters.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'param': [0.1, 0.2, 0.3, 0.4],
         'metric': [0.8, 0.85, 0.9, 0.88]
     })
     sns.lineplot(data=df, x='param', y='metric')
     plt.title('Model Sensitivity')
     plt.savefig('sensitivity_plot.png')
     ```

163. **How do you visualize model prediction errors?**  
     Uses `sns.histplot` to show error distribution.  
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     sns.histplot(data=[0.1, -0.2, 0.15, -0.1, 0.3])
     plt.title('Prediction Error Distribution')
     plt.savefig('prediction_errors.png')
     ```

164. **How do you create a model comparison pairplot?**  
     Uses `sns.pairplot` to compare model predictions.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'Model_A': [0.8, 0.85, 0.9],
         'Model_B': [0.82, 0.87, 0.88],
         'true': [0.81, 0.86, 0.89]
     })
     sns.pairplot(df)
     plt.savefig('model_comparison_pairplot.png')
     ```

165. **How do you visualize model training stability?**  
     Uses `sns.boxplot` to show metric variability across runs.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'run': ['R1', 'R1', 'R2', 'R2'],
         'metric': [0.85, 0.83, 0.84, 0.86]
     })
     sns.boxplot(data=df, x='run', y='metric')
     plt.title('Training Stability')
     plt.savefig('training_stability.png')
     ```

166. **How do you create a feature importance comparison plot?**  
     Uses `sns.catplot` to compare feature importance across models.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'feature': ['F1', 'F2', 'F1', 'F2'],
         'importance': [0.5, 0.3, 0.4, 0.35],
         'model': ['A', 'A', 'B', 'B']
     })
     sns.catplot(data=df, x='feature', y='importance', hue='model', kind='bar')
     plt.title('Feature Importance Comparison')
     plt.savefig('feature_importance_comparison.png')
     ```

167. **How do you visualize model performance trends?**  
     Uses `sns.lineplot` to show metric trends over iterations.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'iteration': [1, 2, 3, 4],
         'metric': [0.75, 0.8, 0.85, 0.9],
         'type': ['test', 'test', 'test', 'test']
     })
     sns.lineplot(data=df, x='iteration', y='metric', hue='type')
     plt.title('Performance Trends')
     plt.savefig('performance_trends.png')
     ```

168. **How do you create a model uncertainty comparison plot?**  
     Uses `sns.kdeplot` to compare uncertainty distributions.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'uncertainty': [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
         'model': ['A', 'A', 'A', 'B', 'B', 'B']
     })
     sns.kdeplot(data=df, x='uncertainty', hue='model')
     plt.title('Uncertainty Comparison')
     plt.savefig('uncertainty_comparison.png')
     ```

169. **How do you visualize model performance by feature range?**  
     Uses `sns.FacetGrid` to show metrics across feature ranges.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'feature': [1, 2, 3, 4, 5, 6],
         'metric': [0.8, 0.85, 0.9, 0.87, 0.83, 0.88],
         'range': ['Low', 'Low', 'Low', 'High', 'High', 'High']
     })
     g = sns.FacetGrid(df, col='range')
     g.map(sns.scatterplot, 'feature', 'metric')
     plt.savefig('performance_by_feature_range.png')
     ```

170. **How do you create a comprehensive model evaluation plot?**  
     Uses `sns.catplot` with multiple metrics and facets.  
     ```python
     import seaborn as sns
     import pandas as pd
     import matplotlib.pyplot as plt
     df = pd.DataFrame({
         'model': ['A', 'A', 'B', 'B'],
         'metric': [0.8, 0.85, 0.9, 0.88],
         'type': ['accuracy', 'precision', 'accuracy', 'precision'],
         'dataset': ['D1', 'D1', 'D2', 'D2']
     })
     sns.catplot(data=df, x='model', y='metric', hue='type', col='dataset', kind='bar')
     plt.savefig('comprehensive_evaluation.png')
     ```