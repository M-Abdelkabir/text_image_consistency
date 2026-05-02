# MVP Détection d'incohérence sémantique texte-image

## Description
Système de détection d'incohérence entre texte et image utilisant le Machine Learning classique (Random Forest) et conteneurisé avec Docker.

## Installation et exécution
1. Téléchargez les datasets :
   ```bash
   python scripts/download_datasets.py --dataset mscoco
   ```
2. Générez les paires :
   ```bash
   python scripts/generate_pairs.py --source data/raw --output data/processed
   ```
3. Entraînez le modèle :
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python src/models/train.py
   ```
4. Lancez avec Docker :
   ```bash
   docker-compose build
   docker-compose up -d
   ```
5. Accédez à l'interface web : http://localhost

## Structure
- `api/` : API Flask et Dockerfile
- `frontend/` : Interface web statique
- `src/` : Code source pour features et modèles
- `scripts/` : Utilitaires de téléchargement et génération

## Licence
MIT
