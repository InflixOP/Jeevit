import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

model = pickle.load(open("gene.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
mlb = pickle.load(open("multi_label_binarizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("gene.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        gene_encoded = label_encoders['name'].transform([data['gene_name']])[0]
        dna_encoded = label_encoders['dna type'].transform([data['dna_type']])[0]
        drug_name = data['drug']

        input_features = np.array([[gene_encoded, dna_encoded]])
        predicted_labels = model.predict(input_features)
        predicted_drugs = mlb.inverse_transform(predicted_labels)[0]

        if drug_name in predicted_drugs:
            result = "Not Suitable"
        else:
            result = "Suitable"

        return jsonify({'Predicted Drugs': predicted_drugs, 'Suitability': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
