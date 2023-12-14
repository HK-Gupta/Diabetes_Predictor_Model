from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello Patient"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the POST request
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        Bmi = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))


        # Create a NumPy array with the input values
        input_query = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness
                                 , Insulin, Bmi, DiabetesPedigreeFunction, Age]])

        # Make predictions
        result = model.predict(input_query)[0]

        # Return the result as JSON
        return jsonify({'Outcome': str(result)})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)