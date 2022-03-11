## Installation de l'environnement virtuel

Créer l'environnement à partir du fichier yaml
```bash
conda env create -f environment.yml
```

Activer l'environnement
```bash
conda activate projet_5
```

Quitter l'environnement
```bash
conda deactivate projet_5
```

Supprimer l'environnement
```bash
conda env remove --name projet_5
```

## Téléchargement du jeu de données

Récupérer les jeux de données <a href = https://www.kaggle.com/olistbr/brazilian-ecommerce>à cette adresse</a>

Récupérer et placer les fichiers csv dans le dossier "data/"

## Utilisation des notebooks

Lancer le 1er notebook "P5_01_notebook_analyse.ipynb" qui sert au merge des données, celui ci va créer un fichier "merge.csv" à la racine.

Lancer les autres notebooks "P5_02_notebook_model.ipynb" et "P5_02_notebook_simulation.ipynb" seulement après avoir crée ce fichier.
