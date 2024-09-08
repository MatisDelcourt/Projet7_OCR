import os
import pickle

import joblib
import pandas as pd
import shap
from flask import Flask, jsonify, request

app = Flask(__name__)

# Récupérez le répertoire actuel du fichier api.py
current_directory = os.path.dirname(os.path.abspath(__file__))

print(current_directory)

# Charger le modèle en dehors de la clause if __name__ == "__main__":
# model_path = os.path.join(current_directory, "..", "Dashboard", "resources", "modele", "best_model_v2.pickle")
model_path = os.path.join(current_directory, "best_model_v2.pickle")
print(model_path)
model = joblib.load(model_path)

# Définition de la route racine qui retourne un message de bienvenue
@app.route("/", methods=["GET"])
def home():
    """ Endpoint racine qui fournit un message de bienvenue. """
    return jsonify({"message": "Bienvenue sur l'API de prédiction du crédit score d'un client"})

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    sk_id_curr = data['SK_ID_CURR']

    # test_path = os.path.join(current_directory, "..", "Dashboard", "resources", "data", "test_set.pickle")
    test_path = os.path.join(current_directory, "test_set.pickle")
    with open(test_path, 'rb') as df_appli_test_set:
        test_set = pickle.load(df_appli_test_set)
    test_set.shape

    df = test_set
    sample = df[df['SK_ID_CURR'] == sk_id_curr]

    # Supprimer la colonne ID pour la prédiction
    sample = sample.drop(columns=['SK_ID_CURR'])

    # Prédire
    prediction = model.predict_proba(sample)
    print(prediction)
    proba = prediction[0][1] # Probabilité de la seconde classe

    # Calculer les valeurs SHAP pour l'échantillon donné
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    if len(shap_values) == 1:
        shap_values_output = shap_values[0][0].tolist()  # Ajustez cette ligne selon la structure
    else:
        shap_values_output = shap_values[1][0].tolist()  # Ajustez si nécessaire

    return jsonify({
        '1-probability': proba * 100,
        '2-shap_values': shap_values_output,
        '3-feature_names': sample.columns.tolist(),
        '4-feature_values': sample.values[0].tolist()
    })

if __name__ == "__main__":
    port = os.environ.get("PORT", 5002)
    app.run(debug=False, host="0.0.0.0", port=int(port))