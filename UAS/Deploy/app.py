from flask import Flask, render_template
from flask import request
import numpy as np
import pandas as pd
import pickle
import joblib

app = Flask(__name__)
model = joblib.load("RFC_113(1.0.2).pkl")

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    feature = [np.array(float_features)]
    prediction = model.predict(feature)
    return render_template("index.html", prediction_text = "{}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)