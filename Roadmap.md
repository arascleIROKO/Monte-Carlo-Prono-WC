🏆 World Cup Prediction Engine
Product Requirements Document (PRD)

Auteur : Anthony Rascle

Version : 1.0

🎯 Objectif

Construire un moteur de prédiction capable de maximiser le nombre de points obtenus dans une compétition de pronostics de Coupe du Monde.

Le moteur ne cherche PAS à prédire le score le plus probable.

Le moteur cherche à maximiser l'espérance de points en fonction du règlement de la compétition.

📋 Règlement

Chaque match rapporte :

Résultat	Points
Bon vainqueur	2
Bon écart de buts	4
Score exact	6

Le score prédit correspond uniquement au temps réglementaire (90 minutes).

🎯 Philosophie

Le projet repose sur quatre principes.

Simple
Robuste
Modulaire
Explicable

On évite toute complexité inutile.

L'objectif n'est pas de construire le modèle de Machine Learning le plus sophistiqué.

L'objectif est de construire le moteur de décision le plus performant.

🏗 Architecture générale
Football API
      │
      ▼
Download Data
      │
      ▼
SQLite Database
      │
      ▼
Elo Rating
      │
      ▼
Expected Goals
      │
      ▼
Poisson Distribution
      │
      ▼
Expected Value Engine
      │
      ▼
Recommendation Engine
      │
      ▼
Dashboard
💻 Stack Technique

Python 3.12

SQLite

Pandas

NumPy

SciPy

Requests

SQLAlchemy

Streamlit

Matplotlib

PyTest

PyYAML

📂 Structure du projet
world-cup-model/

README.md

requirements.txt

config/

    settings.yaml

database/

    database.db

api/

    football_data.py

data/

    raw/

    processed/

models/

    elo.py

    poisson.py

    expected_value.py

    confidence.py

pipeline/

    update.py

dashboard/

    app.py

tests/
⚙ Configuration

Toutes les constantes doivent être modifiables dans :

settings.yaml

Exemple

elo:
    k_factor: 30
    home_advantage: 65

poisson:
    max_goals: 6

simulation:
    iterations: 100000

competition:
    winner_points: 2
    goal_difference_points: 4
    exact_score_points: 6

Aucune constante ne doit être écrite directement dans le code.

📡 API

API officielle utilisée

Football-data.org

Pourquoi ?

gratuite
stable
documentation simple
historique disponible

Toutes les données téléchargées sont sauvegardées dans SQLite.

L'application doit continuer à fonctionner même si l'API devient indisponible.

🗄 Base de données
Teams
id

name

elo

goals_for

goals_against

form

last_update
Matches
id

competition

date

home_team

away_team

home_goals

away_goals

status
Predictions
match_id

prediction

expected_value

confidence

created_at
Results
match_id

real_score

points

winner

goal_difference

exact_score
📈 Pipeline Quotidien

Une seule commande

python pipeline/update.py

Elle devra automatiquement :

1 Télécharger les nouveaux matchs

2 Mettre à jour SQLite

3 Mettre à jour le Elo

4 Calculer les Expected Goals

5 Générer la distribution de Poisson

6 Calculer les Expected Values

7 Générer les recommandations

8 Mettre à jour le dashboard

🧠 Elo Rating

Chaque équipe possède un Elo.

Après chaque résultat :

Ancien Elo

↓

Résultat

↓

Nouveau Elo

Le Elo représente la force intrinsèque d'une équipe.

⚽ Expected Goals

À partir des Elo (et des futures variables),

le moteur calcule

Expected Goals Home

Expected Goals Away

Ces valeurs servent d'entrée au modèle de Poisson.

🎲 Distribution de Poisson

Le moteur calcule automatiquement la probabilité de chaque score.

Exemple

0-0

1-0

2-0

2-1

3-0

3-1

...

6-6

Chaque score possède une probabilité.

💰 Expected Value Engine

Le cœur du projet.

Pour chaque score

Calculer

EV

=

P(score exact) × 6

+

P(écart correct) × 4

+

P(vainqueur correct) × 2

Le moteur renvoie

Top 5 scores
Probabilité
EV

Le score recommandé est celui ayant la plus forte EV.

🎯 Confidence Engine

Chaque recommandation possède un indice de confiance.

Exemple

Brésil 72 %

Nul 18 %

Japon 10 %

Confiance : 91 %

La confiance dépend notamment :

différence Elo
dispersion des probabilités
historique du modèle
📊 Dashboard

Pour chaque match afficher

Brésil vs Japon

Victoire Brésil

72 %

Nul

18 %

Victoire Japon

10 %

----------------

Top Scores

1-0

2-0

2-1

1-1

3-0

----------------

Recommandation

1-0

Confidence

91 %

Expected Value

2.81

Sous le tableau

Historique

Matchs prédits

Exacts

Écarts

Vainqueurs

Moyenne de points

Points totaux
📉 Calibration

Après chaque journée

Comparer

Probabilités prévues

↓

Résultats réels

↓

Erreur

↓

Ajustement automatique

Le modèle s'améliore progressivement.

🚀 Sprints
Sprint 1
Initialiser le projet
SQLite
Configuration
API
Tests
Sprint 2
Téléchargement des données
Stockage
Validation
Sprint 3
Elo complet
Tests
Visualisation
Sprint 4
Expected Goals
Distribution de Poisson
Sprint 5
Expected Value Engine
Recommandations
Sprint 6
Dashboard
Historique
Sprint 7
Calibration automatique
🔮 Version 2

Ajouter progressivement

Classement FIFA
Forme récente
Blessures
Suspensions
Jours de repos
Probabilités Bookmakers
🤖 Version 3

Ajouter un véritable modèle ML.

Le modèle ne prédit jamais directement le score.

Il prédit les Expected Goals.

Ces Expected Goals alimentent ensuite la distribution de Poisson.

Le modèle envisagé est :

LightGBM
ou XGBoost
✅ Règles de développement

Claude Code devra toujours :

écrire un code propre
utiliser le typage Python
documenter chaque fonction
écrire des tests unitaires
éviter les duplications
privilégier les petites fonctions
respecter une architecture modulaire
ne jamais complexifier inutilement le projet

L'objectif est de construire un moteur de prédiction fiable, maintenable et évolutif.