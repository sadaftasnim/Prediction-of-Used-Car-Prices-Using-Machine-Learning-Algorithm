# Import necessary libraries for the Flask app and machine learning
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import load_model  # Uncomment if using Keras model
import joblib  # For loading the trained model

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained RandomForest model (saved as 'car_price_pred_model.pkl')
model = joblib.load('car_price_pred_model.pkl')

# Dictionary for encoding car brands to numeric values
brands = {
    'alfa_romeo': 0, 'audi': 1, 'bmw': 2, 'chevrolet': 3, 'chrysler': 4, 'citroen': 5, 
    'dacia': 6, 'daewoo': 7, 'daihatsu': 8, 'fiat': 9, 'ford': 10, 'honda': 11, 'hyundai': 12, 
    'jaguar': 13, 'jeep': 14, 'kia': 15, 'lada': 16, 'lancia': 17, 'land_rover': 18, 
    'mazda': 19, 'mercedes_benz': 20, 'mini': 21, 'mitsubishi': 22, 'nissan': 23, 'opel': 24, 
    'peugeot': 25, 'porsche': 26, 'renault': 27, 'rover': 28, 'saab': 29, 'seat': 30, 'skoda': 31, 
    'smart': 32, 'sonstige_autos': 33, 'subaru': 34, 'suzuki': 35, 'toyota': 36, 'trabant': 37, 
    'volkswagen': 38, 'volvo': 39
}

# Dictionary for encoding gearbox type to numeric values
gear_box = {'manual': 0, 'automatic': 1}

# Dictionary for encoding fuel type to numeric values
fuel = {'benzin': 0, 'diesel': 1, 'lpg': 2}

# Route to render the homepage
@app.route('/')
def home():
    # Render the HTML template for the homepage
    return render_template('index.html')

# Route to handle prediction requests from the user
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data submitted by the user
        year = int(request.form['year'])  # Year of registration
        gearbox = request.form['gearbox']  # Gearbox type (manual/automatic)
        powerPS = int(request.form['powerPS'])  # Car's power in PS (horsepower)
        kilometer = int(request.form['kilometer'])  # Distance the car has traveled (in km)
        month = int(request.form['month'])  # Month of registration
        fuelType = request.form['fuelType']  # Fuel type (benzin, diesel, etc.)
        brand = request.form['brand']  # Car brand (e.g., audi, bmw, etc.)

        # Create a DataFrame with the input values
        new_data = pd.DataFrame({
            'yearOfRegistration': [year],  # Year of registration
            'gearbox': gear_box[gearbox],  # Convert gearbox to numeric using the dictionary
            'powerPS': [powerPS],  # Power of the car
            'kilometer': [kilometer],  # Distance driven by the car
            'monthOfRegistration': [month],  # Month of registration
            'fuelType': fuel[fuelType],  # Convert fuel type to numeric
            'brand': brands[brand]  # Convert brand to numeric
        })

        # Use the pre-trained model to predict the price based on the input features
        prediction = model.predict(new_data)

        # Render the result on the homepage with the predicted price
        return render_template('index.html', prediction_text=f'Predicted price: {prediction[0]:.2f} EUR')

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
