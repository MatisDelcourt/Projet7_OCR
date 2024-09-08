# Utiliser l'image officielle légère de Python.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Permettre l'affichage immédiat des instructions et messages de log dans les journaux de Knative
ENV PYTHONUNBUFFERED True

# Copier le code local dans l'image du conteneur.
# Définir le répertoire de travail dans le conteneur à /app
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Installer les dépendances de production.
# Exécute pip install pour les packages spécifiés dans requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt

# Exécuter le service web au démarrage du conteneur. Ici, nous utilisons le serveur web gunicorn,
# avec un processus worker et 8 threads.
# Pour des environnements avec plusieurs cœurs de CPU, augmentez le nombre de workers
# pour qu'il soit égal au nombre de cœurs disponibles.
# Le timeout est réglé sur 0 pour désactiver les timeouts des workers pour permettre à Cloud Run de gérer l'échelonnement des instances.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 api:app