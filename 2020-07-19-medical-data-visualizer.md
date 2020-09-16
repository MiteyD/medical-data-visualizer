---
title: "Medical Data Visualizer"
date: 2020-07-19
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---


# Medical Data Visualizer

This project involved visualizing and make calculations from medical examination data using matplotlib, seaborn, and pandas.

The datasets can be found in my github account using the link below.


```python
# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv') 
    
# Add 'overweight' column
height = df.iloc[:, 3]
weight = df.iloc[:, 4]
bmi = pd.DataFrame(weight / (height/100)**2)
bmi = bmi.rename(columns= {0 : 'bmi'})

def f(row):
    if row['bmi'] > 25:
        val = 1
    else:
        val = 0
    return val


bmi['A'] = bmi.apply(f, axis=1)
df['overweight'] = bmi['A']

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
from sklearn.preprocessing import minmax_scale
df = minmax_scale(df, feature_range=(0, 1), axis=0, copy=True)
df = pd.DataFrame(df)
df = df.rename(columns={0: 'id', 1: 'age', 2: 'gender', 3: 'height', 4: 'weight', 
                                        5: 'ap_hi', 6: 'ap_lo', 7: 'cholesterol', 8: 'gluc', 9: 'smoke', 
                                        10: 'alco', 11: 'active', 12: 'cardio', 13: 'overweight'})


def c(row):
    if row['cholesterol'] >= 0.5:
        val = 1
    else:
        val = 0
    return val
df['cholesterol'] = df.apply(c, axis=1)


def g(row):
    if row['gluc'] >= 0.5:
        val = 1
    else:
        val = 0
    return val
df['gluc'] = df.apply(g, axis=1)


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.

    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 
                                                             'overweight'] , var_name='variable', value_name='value')
    
    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the collumns for the catplot to work correctly.

    df_cat = (df_cat.groupby('cardio').filter(lambda x : len(x) > 500).groupby(['cardio', 'variable', 'value']).size().
          to_frame('total').reset_index())
    
    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', y='total', col='cardio', data=df_cat, kind='bar', hue='value', ci=None)
    fig = fig.fig
 
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))
    
    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig

```
