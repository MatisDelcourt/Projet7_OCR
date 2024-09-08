import pytest

# Importe l'application Flask déjà créée
from api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Teste la route racine pour s'assurer qu'elle retourne le message de bienvenue."""
    response = client.get('/')
    assert response.status_code == 200
    assert response.json == {"message": "Bienvenue sur l'API de prédiction du crédit score d'un client"}

def test_predict_route(client):
    """Teste la route de prédiction pour s'assurer qu'elle retourne les résultats attendus."""
    # Prépare des données de test
    test_data = {'SK_ID_CURR': 100001}
    response = client.post('/predict', json=test_data)
    data = response.get_json()
    
    # Vérifie la réponse
    assert response.status_code == 200
    assert '1-probability' in data
    assert '2-shap_values' in data
    assert '3-feature_names' in data
    assert '4-feature_values' in data

def test_model_loading():
    """Teste le chargement correct du modèle."""
    from api import model
    assert model is not None, "Le modèle n'a pas été chargé."

import time
def test_predict_route_performance(client):
    """Teste la performance de la réponse de l'API."""
    start_time = time.time()
    test_data = {'SK_ID_CURR': 100001}
    response = client.post('/predict', json=test_data)
    elapsed_time = time.time() - start_time
    assert elapsed_time < 10

def test_response_format(client):
    """Teste que la réponse de l'API est bien formatée."""
    test_data = {'SK_ID_CURR': 100001}
    response = client.post('/predict', json=test_data)
    data = response.get_json()
    assert isinstance(data['1-probability'], float)
    assert isinstance(data['2-shap_values'], float)
    assert isinstance(data['3-feature_names'], list)
    assert isinstance(data['4-feature_values'], list)

def test_api():
    """Teste que le score prédit par l'API est bien fourni et donne une valeur positive."""
    from flask import json
    with app.test_client() as client:
        # Données de test
        test_data = {'SK_ID_CURR': 100001}
        response = client.post("/predict", json=test_data)
        data = json.loads(response.data)
        prediction = data['1-probability']
        # Vérifie que la prédiction a été effectuée correctement
        assert prediction is not None, "La prédiction a échoué."
        assert prediction >= 0, "La prédiction devrait être une valeur positive."

