import requests

# URL de base de l'API
# url_base = 'http://127.0.0.1:5002'

url_base = 'https://apipred-518395586009.europe-west9.run.app'

# Test du endpoint d'accueil
response = requests.get(f"{url_base}/")
print("Réponse d'accueil:", response.text)
# Données d'exemple pour la prédiction
donnees_predire = {'SK_ID_CURR': 100001}

# Test du endpoint de prédiction
response = requests.post(f"{url_base}/predict", json=donnees_predire)
print("Réponse de prédiction:", response.text)