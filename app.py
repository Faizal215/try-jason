import json
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='template')

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Retrieve user input as a JSON string
        input_str = request.form.get("query")

        # Parse the JSON string into a dictionary
        input_data = json.loads(input_str)

        # Create a DataFrame from the input dictionary
        input_df = pd.DataFrame([input_data])

        # Load the trained model
        model = pickle.load(open("PumpDiagnosticModel.pkl", "rb"))

        # Make a prediction and calculate the probability
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]

        # Determine the pump condition and confidence
        pump_conditions = ["health", "severe", "mild", "unstable"]
        condition = pump_conditions[prediction[0]]
        confidence = probability[0] * 100

        # Render the results on the home page
        output1 = f"The pump is in {condition} condition"
        output2 = f"Confidence: {confidence:.2f}%"

        return render_template('home.html', output1=output1, output2=output2, **request.form)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return render_template('home.html', error_msg=error_msg, **request.form)


if __name__ == "__main__":
    app.run()
