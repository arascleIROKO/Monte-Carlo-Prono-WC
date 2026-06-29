# Prono d'Anto

Moteur de pronostics Coupe du Monde base sur Elo, expected goals, Poisson et expected value.

Le dashboard Streamlit sert a:
- suivre les matchs a venir,
- comparer les predictions du modele et les resultats reels,
- explorer le classement Elo,
- afficher les points Anto issus de `score.md`.

## Stack

- Python
- SQLite
- Streamlit
- NumPy
- SciPy
- SQLAlchemy
- Plotly

## Lancer le projet

Installer les dependances:

```bash
pip install -r requirements.txt
```

Lancer le dashboard:

```bash
streamlit run dashboard/app.py
```

## Pipeline de donnees

Pour synchroniser les donnees, recalculer les ratings Elo et regenerer les predictions:

```bash
python pipeline/update.py
```

## Structure

- `dashboard/` : interface Streamlit
- `pipeline/` : synchronisation et recalcul des donnees
- `models/` : Elo, Poisson, expected value, confiance
- `database/` : schema et base SQLite
- `api/` : client Football-Data
- `config/` : configuration du moteur

## Notes

- Les predictions modeles sont stockees en base.
- `score.md` contient les pronostics Anto affiches dans l'historique.
- La configuration du modele est dans `config/settings.yaml`.
