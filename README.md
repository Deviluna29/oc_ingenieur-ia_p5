## Résultats des Notebooks sous format HTML

Les résultats des notebooks, en HTML avec les graphes interactifs :

<a href = https://deviluna29.github.io/oc_ingenieur-ia_p5/P5_01_notebook_analyse>HTML Notebook analyse et merge</a>

<a href = https://deviluna29.github.io/oc_ingenieur-ia_p5/P5_02_notebook_model.html>HTML Notebook modélisation</a>

<a href = https://deviluna29.github.io/oc_ingenieur-ia_p5/P5_03_notebook_simulation.html>HTML Notebook contrat de maintenance</a>

## Installation de l'environnement virtuel

Créer l'environnement à partir du fichier yaml
```bash
conda env create -f environment.yml
```

Activer l'environnement
```bash
conda activate projet_5
```

## Téléchargement du jeu de données

Récupérer les jeux de données <a href = https://www.kaggle.com/olistbr/brazilian-ecommerce>à cette adresse</a>

Récupérer et placer les fichiers csv dans le dossier "data/"

Il doit y avoir ces fichiers :

- olist_customers_dataset.csv
- olist_geolocation_dataset.csv
- olist_order_items_dataset.csv
- olist_order_payments_dataset.csv
- olist_order_reviews_dataset.csv
- olist_orders_dataset.csv
- olist_products_dataset.csv
- olist_sellers_dataset.csv

## Utilisation des notebooks

Lancer le 1er notebook "P5_01_notebook_analyse.ipynb" qui sert au merge des données, celui ci va créer un fichier "merge.csv" à la racine.

Lancer les autres notebooks "P5_02_notebook_model.ipynb" et "P5_03_notebook_simulation.ipynb" seulement après avoir crée ce fichier.
