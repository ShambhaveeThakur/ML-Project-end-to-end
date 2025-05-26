import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Route to render homepage
@app.route('/')
def home():
    return render_template('home.html')

# API endpoint for direct JSON input prediction
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Input data:", data)
    input_array = np.array(list(data.values())).reshape(1, -1)
    new_data = scalar.transform(input_array)
    prediction = regmodel.predict(new_data)
    return jsonify({'prediction': float(prediction[0])})

# Route to handle form submission prediction
@app.route('/predict', methods=['GET','POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print("Transformed Input:", final_input)
    prediction = regmodel.predict(final_input)
    return render_template("home.html", prediction_text="The predicted price is: {:.2f}".format(prediction[0]))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
