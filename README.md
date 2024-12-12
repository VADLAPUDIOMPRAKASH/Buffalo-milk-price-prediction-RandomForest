# Buffalo Milk Price Prediction Web Application

This project is a Flask-based web application that predicts the price of buffalo milk based on the provided SNF (Solid-Not-Fat) and FAT content, as well as a specific date. The predictions are generated using a pre-trained Random Forest model.

## Features
1. **Milk Price Prediction**: Predicts the price of buffalo milk using SNF, FAT, and the date as inputs.
2. **Prediction History**: Stores all predictions made along with the inputs and timestamps.
3. **Dynamic Model Download**: The model file is not included due to its size but can be downloaded by running the provided Python script.

## Prerequisites
- Python 3.x
- Flask
- pandas
- joblib

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd BUFFALO_MILK_PRICE_PREDICTION-main
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the Random Forest model by running the model downloading script:
   ```bash
   python python_programs/model_downloading_program.py
   ```

5. Ensure the following file paths are set correctly in the `app.py`:
   - Path to the `.pkl` model file.
   - Path to the `predicted.csv` file for storing prediction history.

## Usage
1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open a web browser and navigate to `http://127.0.0.1:5000`.

3. Use the following features:
   - **Prediction**: Enter the SNF, FAT, and date to get a predicted price.
   - **History**: View all past predictions along with their timestamps.

## File Structure
- **app.py**: Main Flask application file.
- **templates/**: Contains HTML templates (`index.html` and `prediction.html`).
- **python_programs/model_downloading_program.py**: Script to download the Random Forest model.
- **predicted.csv**: File for storing the history of predictions.

## Input and Output
### Inputs:
- **Rate Date**: The date for which the prediction is made (YYYY-MM-DD format).
- **SNF**: Solid-Not-Fat percentage in the milk.
- **FAT**: Fat percentage in the milk.

![image](https://github.com/user-attachments/assets/fe0dac2a-9c40-42fa-89fc-edfa7b4dc22d)

![image](https://github.com/user-attachments/assets/c8cf9f1b-c6e4-4e60-a8c3-785f0017f478)


![image](https://github.com/user-attachments/assets/8f6d824f-8562-47e9-8d15-38932091623b)




### Outputs:
- Predicted price of the milk.
- Saved history of predictions in the `predicted.csv` file.

## Notes
1. Ensure the `model_downloading_program.py` script is executed before running the Flask app to obtain the model file.
2. If `predicted.csv` does not exist or is empty, it will be created automatically.

## Dependencies
- Flask
- pandas
- joblib

## Running Tests
1. To test the application locally, provide sample input values on the web interface and verify the outputs.
2. Check the `predicted.csv` file to confirm the storage of prediction history.

## Acknowledgments
Special thanks to the contributors for developing the Random Forest model and the tools necessary for this project.

## License
This project is licensed under the MIT License.

