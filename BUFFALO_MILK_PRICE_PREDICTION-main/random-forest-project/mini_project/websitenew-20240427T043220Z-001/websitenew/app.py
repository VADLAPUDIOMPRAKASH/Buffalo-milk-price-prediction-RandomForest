from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\KGRCET\Downloads\BUFFALO_MILK_PRICE_PREDICTION-main\rf_regressor_model.pkl')

# Function to predict rate
def predict_rate(rate_date, snf, fat):
    rate_date = datetime.strptime(rate_date, '%Y-%m-%d')
    reference_date = datetime(2000, 1, 1)
    days_since_ref_date = (rate_date - reference_date).days
    
    input_data = pd.DataFrame({
        'DaysSinceRefDate': [days_since_ref_date],
        'SNF': [snf],
        'FAT': [fat]
    })
    
    prediction = model.predict(input_data)
    return prediction[0]

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    rate_date = request.form['RateDate']
    snf = float(request.form['SNF'])
    fat = float(request.form['FAT'])
    
    predicted_rate = predict_rate(rate_date, snf, fat)
    
    # Capture the current date and time
    prediction_time = datetime.now()
    
    # Create a DataFrame to store the prediction and prediction time
    prediction_data = {
        'RateDate': [rate_date],
        'SNF': [snf],
        'FAT': [fat],
        'Rate': [predicted_rate],
        'PredictionTime': [prediction_time]  # Include the prediction time
    }
    
    df_prediction = pd.DataFrame(prediction_data)
    
    try:
        # Load the original dataset
        df = pd.read_csv(r'C:\Users\KGRCET\Downloads\BUFFALO_MILK_PRICE_PREDICTION-main\BUFFALO_MILK_PRICE_PREDICTION-main\random-forest-project\mini_project\websitenew-20240427T043220Z-001\websitenew\predicted.csv')  
    except pd.errors.EmptyDataError:
        # If the file is empty, initialize it with headers
        df = pd.DataFrame(columns=['RateDate', 'SNF', 'FAT', 'Rate', 'PredictionTime'])
    
    # Merge the original dataset with the new prediction data
    df_merged = pd.concat([df, df_prediction], ignore_index=True)
    
    # Save the merged dataset back to CSV
    df_merged.to_csv(r'C:\Users\KGRCET\Downloads\BUFFALO_MILK_PRICE_PREDICTION-main\BUFFALO_MILK_PRICE_PREDICTION-main\random-forest-project\mini_project\websitenew-20240427T043220Z-001\websitenew\predicted.csv', index=False)  
    
    return render_template('index.html', prediction=predicted_rate)

# Route for prediction page
@app.route('/prediction')
def prediction():
    # Load the predicted history data from the CSV file
    try:
        predicted_data = pd.read_csv(r'C:\Users\KGRCET\Downloads\BUFFALO_MILK_PRICE_PREDICTION-main\BUFFALO_MILK_PRICE_PREDICTION-main\random-forest-project\mini_project\websitenew-20240427T043220Z-001\websitenew\predicted.csv')
    except pd.errors.EmptyDataError:
        # If the file is empty or doesn't exist, return an empty DataFrame
        predicted_data = pd.DataFrame(columns=['RateDate', 'SNF', 'FAT', 'Rate', 'PredictionTime'])
    
    # Pass the predicted history data to the prediction.html template
    return render_template('prediction.html', predicted_data=predicted_data)

if __name__ == '__main__':
    app.run(debug=True)
