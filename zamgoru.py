"""
Energy Data Analysis and Forecasting

This code performs analysis and forecasting on energy data using
Python libraries such as Pandas, Seaborn, Matplotlib, and Prophet.
The code follows a step-by-step approach:

Data Cleaning: The provided CSV file is read into a Pandas
DataFrame. The columns are renamed, irrelevant data is removed,
and missing values are filled using the forward fill method.
Product names are extracted using regular expressions.

Correlation Analysis: A correlation heatmap is generated using
the pivot table of energy prices. The heatmap helps visualize
the relationships between different energy products over time.

Time Series Analysis: The code focuses on the "Electricity Price"
product. A time series plot is created to display the electricity
prices over a specific period.

Forecasting with Prophet: The Prophet library is used to forecast
future electricity prices. The DataFrame is prepared with columns
 named 'ds' (representing dates) and 'y' (representing prices).
 The Prophet model is trained and country-specific holidays (Turkey)
  are added. Future predictions are made for the next 120 months
  (10 years).

The code serves as a starting point for analyzing energy data,
exploring correlations, visualizing time series, and performing
forecasting using the Prophet library.

"""

import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from fbprophet import Prophet

# Read the CSV file
df = pd.read_csv('pivotenergy2.csv', sep="|", header=None, skiprows=4)

## DATA CLEANING ##

# Rename columns
cols = ['Yıl', 'Ay', 'Ürün', 'Fiyat', 'Ekstra']
df.columns = cols

# Remove useless data column
df.drop(axis=1, columns=['Ekstra'], inplace=True)

# Fill NA cells with forward fill method
df = df.fillna(method='ffill')

# Extract product name using regex pattern
pattern = r'.*\((.*)\)+'
df['Ürün'] = df['Ürün'].apply(lambda x: re.search(pattern, x).group(1))
df['Ürün'] = df['Ürün'].apply(lambda x: x.replace(')', ''))

# Convert data types
df['Yıl'] = df['Yıl'].astype(int)
df['Ay'] = df['Ay'].apply(lambda x: int(str(x)[:2]))

### CREATE CORRELATION HEATMAP ###

# Create pivot table
df_pivot = df.pivot_table('Fiyat', ['Yıl', 'Ay'], 'Ürün')

# Calculate correlation
corr = df_pivot.corr()

# Set seaborn visual settings
sns.set(rc={'figure.figsize':(11.7, 8.27)})

# Create correlation heatmap
ax = sns.heatmap(
    corr,
    cmap='Blues',
    vmax=1,
    vmin=0,
    center=0.5,
    annot=True
)
ax.set_xticklabels(corr.index, horizontalalignment='right', rotation=45)

# Sort correlation values
stack = corr.unstack().sort_values(kind='quicksort', ascending=False)

# Display the heatmap
plt.show()

## TIME SERIES ANALYSIS ##

# Define a function to map date values
def date_mapping(row):
    return datetime.date(row['Yıl'], row['Ay'], 1)

# Filter data for controlling electric prices
df = df.loc[df['Ürün'] == 'Elektrik Ücreti']

# Apply date mapping function to create Date column
df['Date'] = df.apply(lambda x: date_mapping(x), axis=1)
df_sub = df[['Date', 'Fiyat']].reset_index(drop=True)

# Plot time series for a certain period
plt.style.use('fivethirtyeight')
pd.plotting.register_matplotlib_converters()
fig = plt.figure(figsize=(10, 6))
df_sub2 = df[df['Yıl'] >= 2018].reset_index(drop=True)
df_sub2['Date'] = df_sub2.apply(lambda x: date_mapping(x), axis=1)
plt.plot(df_sub2['Date'], df_sub2['Fiyat'])
plt.legend(['Fiyat'])
plt.show()

## PROPHET FORECASTING ##

# Prepare data for Prophet model
df_sub.columns = ['ds', 'y']

# Create Prophet model
m = Prophet(
    seasonality_mode='multiplicative',
    interval_width=0.95,
    mcmc_samples=300
)

# Add country holidays (Turkey)
try:
    m.add_country_holidays(country_name='TR')
except:
    None

# Fit the model
m = m.fit(df_sub)

# Make future predictions
future = m.make_future_dataframe(periods=120, freq='M')
fcst = m

plt1=m.plot_components(fcst)

