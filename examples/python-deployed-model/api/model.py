from flask import Flask, jsonify, request
from sklearn.externals import joblib
import numpy as np

estimator = joblib.load('../models/model.pkl')

def json_to_model_input(request_body):
    json_ = request_body.json
    input = json_['input']
    query = np.array(input)
    if len(query.shape) == 1:
        query = query[:, np.newaxis].T
    return query

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Note:
    # This is not a scalable solution
    # Refer to DataScience Inc.'s product offering for a scalable solution
    print request.get_json()
    query = json_to_model_input(request)
    print query.shape
    prediction = estimator.predict(query)
    print prediction
    return jsonify({'predictions': list(prediction)})

if __name__ == '__main__':
     app.run("0.0.0.0", debug=True)

