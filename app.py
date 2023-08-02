# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained ML model
model = joblib.load('query_to_feature_vector.py')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

