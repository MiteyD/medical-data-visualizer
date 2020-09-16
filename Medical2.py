# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 23:54:12 2020

@author: HP
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2) > 25

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
medical_dict = { 1: 0, 2 : 1, 3: 1}
df['cholesterol'] = df['cholesterol'].map( medical_dict )
df['gluc'] = df['gluc'].map( medical_dict ) 

# Draw Cat Plot
def draw_cat_plot():
  # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
  df_cat = pd.melt(
      frame=df, value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], 
      id_vars=['cardio']
  )

  # Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
  df_cat = pd.DataFrame(
      df_cat.groupby(
          ['variable', 'value', 'cardio'])['value'].count()).rename(
          columns={'value': 'total'}).reset_index()

  # Set up the matplotlib figure and draw the catplot
  sns.catplot(x='variable', y='total', hue='value', 
              col='cardio', data=df_cat, kind='bar')

  plt.savefig('catplot.png')
  return plt.gcf()


# Draw Heat Map
def draw_heat_map():
  # Clean the data
  df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
            (df['height'] >= df['height'].quantile(0.025)) &
            (df['height'] <= df['height'].quantile(0.975)) &
            (df['weight'] >= df['weight'].quantile(0.025)) & 
            (df['weight'] <= df['weight'].quantile(0.975))]

  # Calculate the correlation matrix
  corr = df_heat.corr()

  # Generate a mask for the upper triangle
  mask = np.zeros_like(corr, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True

  # Set up the matplotlib figure
  fig, ax = plt.subplots(figsize=(11, 9))

  # Draw the heatmap with the mask
  sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, vmax=.3, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .5})
  fig.savefig('heatmap.png')
  return fig