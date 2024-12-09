import pandas as pd

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')

# View column names in the dataset
df.columns

# Check the value counts for the 'gearbox' column
df['gearbox'].value_counts()

# Map gearbox values to numerical format: manual=0, automatic=1
gearbox = {'manual': 0, 'automatic': 1}

# Check the value counts for the 'brand' column
df['brand'].value_counts()

# Import LabelEncoder to convert text values to numerical format
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Encode the 'brand' column
df['brand'] = le.fit_transform(df['brand'])

# Create a dictionary to map original brand names to their encoded values
brands = {}
for value, label in zip(le.classes_, le.transform(le.classes_)):
    brands[value] = label

# Check the value counts for the 'fuelType' column
df['fuelType'].value_counts()

# Remove uncommon fuel types to focus on the main categories
df = df[df['fuelType'] != 'hybrid']
df = df[df['fuelType'] != 'andere']
df = df[df['fuelType'] != 'elektro']
df = df[df['fuelType'] != 'cng']

# Check the remaining value counts for 'fuelType'
df['fuelType'].value_counts()

# Map fuel types to numerical values: benzin=0, diesel=1, lpg=2
fuel = {'benzin': 0, 'diesel': 1, 'lpg': 2}

# Check data types of all columns
df.dtypes

# Encode 'gearbox' and 'fuelType' columns using the defined mappings
df['gearbox'] = df['gearbox'].map(gearbox)
df['fuelType'] = df['fuelType'].map(fuel)

# Drop any rows with missing values
df = df.dropna()

# Separate the dataset into features (X) and target variable (y)
X = df.drop(columns=['price'])
y = df['price']

# Split data into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import machine learning models and evaluation metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define models to compare: Linear Regression and Random Forest
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions on test set
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Print metrics
    print(f'{name}:')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared: {r2:.2f}')
    print('')

# Choose the best-performing model (Random Forest in this case)
best_model = models['Random Forest']

# Predict the price of a car based on given features
# Example input: Volkswagen car details
cartype = 'volkswagen'  # Car brand
arr = [[brands[cartype]]]  # Encode car brand
prediction = best_model.predict([[2015, 0, 120, 80000, 6, 4, 12]])  # Provide sample car details
print(f'Predicted price: {prediction[0]:.2f}')

# Save the trained model to a file for later use
import joblib
joblib.dump(best_model, 'car_price_pred_model.pkl')

# Output dictionaries for reference
brands  # Encoded brand values
gearbox  # Gearbox mapping
fuel  # Fuel type mapping
