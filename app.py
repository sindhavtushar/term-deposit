from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from data_handling import predict_subscription

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Home Page

@app.route('/form')
def show_form():
    return render_template('form.html')  # Form Page

@app.route('/result')
def show_result():
    return render_template('result.html')  # Result Page

@app.route('/dataset_info')
def dataset_info():
    return render_template('dataset_info.html')  # Dataset Info Page

@app.route('/eda')
def eda():
    return render_template('EDA.html')  # EDA Page

@app.route('/about')
def about():
    return render_template('about.html')  # About Page

@app.route('/submit_form', methods=['POST'])
def handle_prediction():
    if request.method == 'POST':
        user_input = request.form.to_dict()
        prediction_result, probability = predict_subscription(user_input)

        return render_template('result.html', 
                               prediction=prediction_result, 
                               probability=probability, 
                               user_input=user_input)


if __name__ == '__main__':
    app.run(debug=True)
