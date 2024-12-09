# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset from the CSV file
df = pd.read_csv('C:/Users/USER/Desktop/car_price/car_price/autos.csv')

# Display the first 10 rows of the dataset
df.head(10)

# Define columns to drop that are not needed for the analysis
columns_to_drop = ['index', 'dateCrawled', 'name', 'seller', 'model', 'abtest', 'notRepairedDamage',
                   'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen']

# Drop the specified columns from the dataset
df.drop(columns=columns_to_drop, inplace=True)

# Display the first 10 rows after dropping the columns
df.head(10)

# Check unique values in the 'offerType' column to understand its categories
df['offerType'].unique()

# Print all the column names in the dataset
print(df.columns)

# Define columns 'offerType' and 'vehicleType' to be dropped as they are not needed
cols = ['offerType', 'vehicleType']

# Drop the 'offerType' and 'vehicleType' columns from the dataset
df.drop(columns=cols, inplace=True)

# Display the first few rows after dropping 'offerType' and 'vehicleType'
df.head()

# Check unique values in the 'fuelType' column
df['fuelType'].unique()

# Check for missing values in the dataset
df.isna().sum()

# Drop rows with missing values
df.dropna(inplace=True)

# Verify that there are no more missing values
df.isna().sum()

# Check unique values in 'fuelType' after handling missing data
df['fuelType'].unique()

# Check unique values in the 'brand' column
df['brand'].unique()

# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_data.csv', index=False)

# Re-read the cleaned data for further analysis
df = pd.read_csv('C:/Users/USER/Desktop/car_price/car_price/cleaned_data.csv')

# Plot a countplot to visualize the distribution of fuel types
plt.figure(figsize=(10, 6))
sns.countplot(x='fuelType', data=df)
plt.title('Distribution of Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.show()

# Convert 'yearOfRegistration' and 'price' columns to numeric, coercing errors to NaN
df['yearOfRegistration'] = pd.to_numeric(df['yearOfRegistration'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Scatter plot to show the relationship between 'yearOfRegistration' and 'price'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='yearOfRegistration', y='price', data=df)
plt.title('Scatter plot of Price vs Year of Registration')
plt.xlabel('Year of Registration')
plt.ylabel('Price')
plt.show()

# Boxplot to visualize price distribution based on fuel type
plt.figure(figsize=(10, 6))
sns.boxplot(x='fuelType', y='price', data=df)
plt.title('Box plot of Price by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Price')
plt.show()

# Line plot to observe the trend of price over the years of registration
plt.figure(figsize=(10, 6))
sns.lineplot(x='yearOfRegistration', y='price', data=df)
plt.title('Time Series of Price over Year of Registration')
plt.xlabel('Year of Registration')
plt.ylabel('Price')
plt.show()

# Encode categorical columns 'brand', 'fuelType', and 'gearbox' into numerical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Apply label encoding to the 'brand' column
df['brand'] = le.fit_transform(df['brand'])

# Apply label encoding to the 'fuelType' column
df['fuelType'] = le.fit_transform(df['fuelType'])

# Apply label encoding to the 'gearbox' column
df['gearbox'] = le.fit_transform(df['gearbox'])
