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
    return render_template('home.html', output1="", output2="", input_str="")

@app.route("/predict", methods=['POST'])
def predict():
    # Load the trained model
    model = pickle.load(open("PumpDiagnosticModel.pkl", "rb"))

    # Retrieve and parse input data
    input_str = request.form['input_str']
    try:
        input_data = [float(x.strip()) for x in input_str.strip('{}').split(',')]
    except ValueError:
        return render_template('home.html', output1="Invalid input", output2="", input_str=input_str)

    # Check that the input data has the correct number of elements
    if len(input_data) != 5:
        return render_template('home.html', output1="Invalid input", output2="", input_str=input_str)

    # Create a DataFrame from input data
    new_df = pd.DataFrame([input_data], columns=['mean', 'max', 'kurtosis', 'variance', 'onenorm'])

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

    return render_template('home.html', output1=output1, output2=output2, input_str=input_str)

if __name__ == "__main__":
    app.run()
