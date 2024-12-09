# Car Price Prediction Using Machine Learning

This project uses machine learning techniques to predict the price of used cars based on various features such as the car's brand, year of registration, gearbox type, fuel type, power, mileage, and more. The model is built using a Random Forest Regressor and deployed as a web application using Flask.

## Project Overview

The goal of this project is to predict the price of a used car based on the following factors:
- **Year of Registration**: The year the car was registered.
- **Gearbox**: The type of gearbox in the car (manual or automatic).
- **Power (PS)**: The car's horsepower.
- **Kilometer**: The number of kilometers the car has been driven.
- **Month of Registration**: The month when the car was registered.
- **Fuel Type**: The fuel type of the car (e.g., benzine, diesel).
- **Brand**: The car's brand (e.g., Audi, BMW, Ford).

## Technologies Used

- **Python 3.7**: Programming language used to build the machine learning model and the Flask application.
- **Flask**: Web framework to serve the car price prediction model.
- **Pandas**: Data manipulation and analysis library.
- **Scikit-Learn**: Machine learning library used for model building and evaluation.
- **Joblib**: To save and load the trained model.
- **Matplotlib/Seaborn**: Data visualization libraries.

## Prerequisites

To run this project, you need the following software:

- **Python 3.7** or higher
- **pip** (Python package installer)

You also need to install the following Python libraries:

- **Flask**: Web framework to build and run the web app.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For machine learning algorithms and evaluation.
- **Joblib**: For saving and loading machine learning models.
- **Matplotlib**: For data visualization (optional, used for plotting).
- **Seaborn**: For advanced data visualization (optional, used for plotting).

### To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

### The `requirements.txt` file should include the following:

```
Flask==2.1.1
pandas==1.5.3
scikit-learn==1.2.2
joblib==1.1.1
matplotlib==3.7.1
seaborn==0.12.2
```

## How the Project Works

1. **Data Preprocessing**:
    - The dataset is cleaned by removing unnecessary columns, handling missing data, and encoding categorical variables (e.g., brand, gearbox, fuel type) into numerical values.
    - The cleaned data is used to train a Random Forest model that predicts the price of a car based on input features.

2. **Model Training**:
    - A **Random Forest Regressor** is used to train the model on the preprocessed data.
    - The model is then saved using `joblib` for later use in the Flask app.

3. **Flask Web Application**:
    - The web application allows users to input car features (e.g., brand, year of registration, etc.) and predicts the price of the car.
    - The Flask app loads the trained model and uses it to make predictions based on user input.

4. **Deployment**:
    - The app is deployed locally and can be accessed in a browser.
    - The user inputs the car's details, and the app returns the predicted price.

## Installation

Follow these steps to run the project locally:

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Install Dependencies

Create a virtual environment and install the required libraries:

```bash
python -m venv venv
source venv/bin/activate  # For Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download the dataset (`autos.csv`) and save it in the project folder. Ensure it is cleaned and preprocessed as per the project.

### 4. Train the Model

Run the `train_model.py` script to train the Random Forest Regressor model and save it as `car_price_pred_model.pkl`.

```bash
python train_model.py
```

### 5. Run the Flask App

To start the Flask application:

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser to interact with the car price prediction web app.

## Usage

1. Navigate to the home page where you will find a form to input details about the car.
2. Input the following details:
    - **Year of Registration**: The year the car was registered.
    - **Gearbox**: Choose between manual and automatic.
    - **PowerPS**: Enter the car's horsepower (PS).
    - **Kilometer**: Enter the distance driven by the car.
    - **Month of Registration**: The month when the car was registered.
    - **Fuel Type**: Select the fuel type (benzin, diesel, etc.).
    - **Brand**: Select the car's brand (e.g., Audi, BMW).
3. Once the form is submitted, the model will predict the car's price based on the input features.

## Project Structure

```
/car_price_prediction
    ├── app.py                                # Flask application to handle user input and predictions
    ├── car_price_pred_model.pkl              # Saved model file (generated from Selecting_And_dumping_Model.py)
    ├── autos.csv                             # Original dataset used for cleaning and training
    ├── cleaned_data.csv                      # Cleaned dataset used for training (generated from Cleaning_And_Visualizing_CSV_Data.py)
    ├── Cleaning_And_Visualizing_CSV_Data.py  # Script to clean and visualize the data
    ├── Selecting_And_dumping_Model.py        # Script to train and save the model
    ├── templates/
    │   └── index.html                        # HTML file for the front-end user interface
    ├── static/
    │   └── background.jpg                    # Background image for the webpage
    ├── requirements.txt                      # Required Python packages
    └── README.md                             # Project documentation



## Evaluation Metrics

The model is evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: Measures the average of the absolute errors between predicted and actual values.
- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors.
- **R-squared (R²)**: Measures how well the model explains the variance in the data.

## Future Improvements

- **Add more features**: Additional features like car color, condition, or previous owner count could improve prediction accuracy.
- **Model Optimization**: Hyperparameter tuning and cross-validation could be implemented for better model performance.
- **Deployment on Cloud**: Deploy the Flask app to a cloud platform such as Heroku or AWS for wider accessibility.