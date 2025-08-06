from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp = float(request.form['temperature'])
        prediction = model.predict([[temp]])
        predicted_revenue = round(prediction[0][0], 2)
        return render_template('index.html', 
                               prediction_text=f"Predicted Revenue for {temp}°C is ${predicted_revenue}")
    except:
        return render_template('index.html', 
                               prediction_text="Invalid input! Please enter a valid number.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
