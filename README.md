# Estatia House Price Prediction

Application de prédiction du prix des maisons avec Python, scikit-learn et Streamlit.
Le projet nettoie les données immobilières, crée des variables utiles, compare plusieurs modèles et expose le meilleur pipeline via une interface web.

## Fonctionnalités

- Nettoyage des données et traitement des valeurs manquantes et aberrantes
- Génération des données Silver et Gold
- Création de variables pour l'entraînement
- Comparaison des modèles Linear, Non-Linear et Random Forest
- Optimisation des hyperparamètres par validation croisée
- Sauvegarde du meilleur pipeline avec Joblib
- Application Streamlit pour estimer le prix d'un logement
- Visualisation de la prédiction sur la distribution des prix

## Structure du projet

```text
data/bronze/    Données sources
data/silver/    Données nettoyées
data/gold/      Données transformées pour le modèle
src/            Extraction, nettoyage et chargement des données
features/       Création des variables
models/         Modèles et pipeline sauvegardé
evaluation/     Évaluation et optimisation des modèles
dashboard/      Interface Streamlit
```

## Installation locale

Python 3.11 ou une version compatible est recommandé.

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit joblib matplotlib numpy pandas rich scikit-learn seaborn
```

## Préparer les données et entraîner le modèle

Depuis la racine du projet :

```bash
python main.py
```

Cette commande lit `data/bronze/House_Prices.csv`, génère les fichiers Silver et Gold, compare les modèles et sauvegarde le meilleur pipeline dans `models/house_price_model.joblib`.

## Lancer le dashboard

Le fichier du modèle doit être présent avant le démarrage de l'application.

```bash
streamlit run dashboard/app.py
```

L'application est ensuite accessible à l'adresse [http://localhost:8501](http://localhost:8501).

## Lancer avec Docker Compose

```bash
docker compose up --build
```

Le dashboard est disponible sur [http://localhost:8501](http://localhost:8501). Pour utiliser un autre port local :

```bash
APP_PORT=8502 docker compose up --build
```

Pour arrêter le service :

```bash
docker compose down
```

## Données et modèle

- Source : `data/bronze/House_Prices.csv`
- Données nettoyées : `data/silver/House_Prices_Silver.csv`
- Données transformées : `data/gold/House_Prices_Gold.csv`
- Pipeline entraîné : `models/house_price_model.joblib`

Le modèle enregistré contient les transformations nécessaires aux variables numériques et catégorielles. Le formulaire Streamlit utilise les colonnes des données Gold pour construire une prédiction.