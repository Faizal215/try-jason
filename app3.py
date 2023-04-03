import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='template')

@app.route("/")
def loadPage():
    return render_template('home.html', output1="", output2="", tsv_input="")

@app.route("/predict", methods=['POST'])
def predict():
    # Load the trained model
    model = pickle.load(open("PumpDiagnosticModel.pkl", "rb"))

    # Retrieve input values
    input_values = request.form['tsv_input'].strip().split('\t')

    # Check that the input values are valid
    if len(input_values) != 5:
        return render_template('home.html', output1="Error: Input must contain exactly 5 values separated by tabs.", output2="", tsv_input=request.form['tsv_input'])
    try:
        data = {"mean": float(input_values[0]),
                "max": float(input_values[1]),
                "kurtosis": float(input_values[2]),
                "variance": float(input_values[3]),
                "onenorm": float(input_values[4])}
    except ValueError:
        return render_template('home.html', output1="Error: Input values must be numbers separated by tabs.", output2="", tsv_input=request.form['tsv_input'])

    new_df = pd.DataFrame(data, index=[0])

    # Make a prediction and calculate the probability
    prediction = model.predict(new_df)
    probability = model.predict_proba(new_df)[:, 1]

    # Determine the pump condition and confidence
    pump_conditions = ["health", "severe", "mild", "unstable"]
    condition = pump_conditions[prediction[0]]
    confidence = probability[0] * 100

    # Render the results on the home page
    output1 = f"The pump is in {condition} condition"
    output2 = f"Confidence: {confidence}"

    return render_template('home.html', output1=output1, output2=output2, tsv_input=request.form['tsv_input'])

if __name__ == "__main__":
    app.run()
