from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import calendar

app = Flask(__name__)

# Load the pre-trained XGBoost model
with open('xgboost_model_.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json
    start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
    category = data['category']
    
    print("Received prediction request from:", start_date, "to:", end_date)
    print("Selected category:", category)

    # Calculate the number of days between start and end date
    num_days = (end_date - start_date).days + 1

    # Process the input data for the selected period
    new_sample = pd.DataFrame({
        'ID': [500] * num_days,
        'Category_M01AB': [0] * num_days,
        'Category_M01AE': [0] * num_days,
        'Category_N02BA': [0] * num_days,
        'Category_N02BE': [0] * num_days,
        'Category_N05B': [0] * num_days,
        'Category_N05C': [0] * num_days,
        'Category_R03': [0] * num_days,
        'Category_R06': [0] * num_days
    })

    # Set the value corresponding to the selected category to 1 for each day
    new_sample[f'Category_{category}'] = 1

    # Make predictions for the selected period
    predictions = model.predict(new_sample)
    
    # Aggregate the predicted quantities for the period
    total_predicted_quantity = predictions.sum()

    print("Total predicted quantity:", total_predicted_quantity)

    # Return the total predicted quantity
    return jsonify({'total_predicted_quantity': float(total_predicted_quantity)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
