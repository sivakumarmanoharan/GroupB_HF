# Dependencies
import sys

from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    lr = joblib.load("nb_model.pkl")  # Load "model.pkl"
    print('Model loaded')
    model_columns = joblib.load("model_columns.pkl")  # Load "model_columns.pkl"
    print('Model columns loaded')
    if lr:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = (lr.predict(query))
            prediction = [int(x) for x in prediction]
            return jsonify({"prediction": prediction[0]})
        except:
            print(traceback.format_exc())
            return jsonify({'trace': traceback.format_exc()}),500
    else:
        print('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    app.run(port=5000, debug=True)
