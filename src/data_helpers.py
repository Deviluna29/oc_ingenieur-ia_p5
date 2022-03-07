import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn import cluster, metrics, decomposition, preprocessing
from sklearn.base import ClusterMixin
from sklearn.manifold import TSNE

import plotly.express as px
import plotly.graph_objects as go

def modelCluster(data):
    #Standardiser les données
    std_scale = preprocessing.StandardScaler().fit(data)
    data_std = pd.DataFrame(
        std_scale.transform(data),
        columns=data.columns
    )

    ### Méthodes des KMeans ###
    # Méthode du coude
    model = cluster.KMeans()
    visualizer_elbow = KElbowVisualizer(model, k=(2, 15))
    visualizer_elbow.fit(data_std)
    visualizer_elbow.show()

    k = visualizer_elbow.elbow_value_

    print(f'Le nombre optimal de clusters est {k}')

    # Méthode du score Silhouettes 
    model = cluster.KMeans(k)
    visualizer_silhouette = SilhouetteVisualizer(model)

    visualizer_silhouette.fit(data_std)
    visualizer_silhouette.poof()

    ### Visualisation ####
    # Predict labels
    model.fit(data_std)
    predicted_labels = model.predict(data_std)
    data["labels"] = predicted_labels

    # Nombre de clients par cluster
    y = data['labels'].value_counts()

    plt.bar(y.index, y, 0.5)
    plt.xlabel("Cluster")
    plt.ylabel("nbr de clients")
    plt.title(f"Histogramme - nbr clients par cluster")
    plt.show()

    for i in range(y.size):
        print(f'Cluster {i}: {y[i]} clients | {round(y[i]*100/y.sum(), 2)}% du total')

    # Boxplot et distribution des features pour chaque cluster
    for column in data.columns:
        if column != 'labels':
            fig = px.box(
                    data,
                    title=f"kMeans - {column}",
                    x=column,
                    color="labels",
                    width=800,
                )
            fig.show()

    # Radar Chart
    data_mean_by_cluster = data.groupby('labels').agg('mean')
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(1, 5))
    data_mean_by_cluster_scaled = minmax_scale.fit_transform(data_mean_by_cluster)

    features = data_mean_by_cluster.columns

    fig = go.Figure()

    for i in range (data_mean_by_cluster.shape[0]):
        fig.add_trace(go.Scatterpolar(
        r=data_mean_by_cluster_scaled[i],
        theta=features,
        fill='toself',
        name=f'Cluster {i}'
    ))

    fig.show()

    # TSNE
    data_trans = TSNE(n_components=2,n_iter=1000,perplexity=30).fit_transform(data_std)

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot()

    scatter = ax.scatter(data_trans[:,0], data_trans[:,1], c=data["labels"],cmap="viridis",alpha=0.3)

    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    legend_1 = ax.legend(handles, labels, loc="upper right", title="Cluster")
    ax.add_artist(legend_1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.show()


# Affiche les histogrammes et les boîtes à moustaches de chaque variable quantitative
def drawHistAndBoxPlot(data, columns, dims_fig):
    nbr_rows = int(len(columns))
    index = 1
    plt.figure(figsize=dims_fig)
    for column in columns:

        log = False if column == "recence" else True

        plt.subplot(nbr_rows, 2, index)
        plt.hist(data[column], log=log, bins=200)
        plt.xlabel(f"{column}")
        plt.ylabel("Count")
        plt.title(f"Histogramme - {column}")
        index += 1

        plt.subplot(nbr_rows, 2, index)
        sns.boxplot(x=data[column])
        plt.xlabel(column)
        plt.title(f"Boite à moustaches pour {column}")
        index += 1
    plt.show()

# Détermine les composantes principales
# Trace les éboulis des valeurs propres
# Affiche la projection des individus sur les différents plans factoriels
# Affiche les cercles de corrélations
def acpAnalysis(data):
    n = data.shape[0]
    p = data.shape[1]

    # On instancie l'object ACP
    acp = decomposition.PCA(svd_solver='full')
    # On récupère les coordonnées factorielles Fik pour chaque individu (projection des individus sur les composantes principales)
    coord = acp.fit_transform(data)

    # Création d'un Datframe contenant les coordonnées factiorelles, le nom de chaque produit et le nom des colonnes correspondant à chaque composante principale
    projected_data = pd.DataFrame(
        data=coord,
        index=data.index,
        columns=[ f'F{i}' for i in range(1, p+1) ]
    )

    # valeur de la variance corrigée
    eigval = (n-1)/n*acp.explained_variance_
    eigval_ratio = (n-1)/n*acp.explained_variance_ratio_

    # On affiche l'éboulis des valeurs propres
    plt.plot(np.arange(1,p+1),eigval,c="red",marker='o')
    plt.title("Eboulis des valeurs propres")
    plt.ylabel("Valeur propre")
    plt.xlabel("Rang de l'axe d'inertie")
    plt.show()

    plt.bar(np.arange(1, p+1), eigval_ratio*100)
    plt.plot(np.arange(1, p+1), eigval_ratio.cumsum()*100,c="red",marker='o')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

    # On détermine le nombre de composantes à analyser
    # On ne considère pas comme importants les axes dont l'inertie associée est inférieue à (100/p)%
    k = np.where(eigval_ratio < 1/p)[0][0]

    pas = 2

    if (k % 2) != 0:
        k -= 1
    
    if k == 0 :
        k = p-1
        pas = 1

    print (f"Le nombre de composantes à analyser est de {k}")

    # racine carrée des valeurs propres
    sqrt_eigval = np.sqrt(eigval)

    # corrélation des variables avec les axes
    corvar = np.zeros((p,p))
    for j in range(p):
        corvar[:,j] = acp.components_[j,:] * sqrt_eigval[j]

    for i in range(0, k, pas):
        # --------------- projection des individus dans un plan factoriel ---------------
        fig, axes = plt.subplots(1, 2, figsize=(24,12))
        axes[0].set_xlim(projected_data[f'F{i+1}'].min(),projected_data[f'F{i+1}'].max())
        axes[0].set_ylim(projected_data[f'F{i+2}'].min(),projected_data[f'F{i+2}'].max())
        sns.scatterplot(
            ax=axes[0], x=f'F{i+1}',
            y=f'F{i+2}',
            data=projected_data
            )

        axes[0].set_xlabel(f'F{i+1}')
        axes[0].set_ylabel(f'F{i+2}')
        axes[0].set_title(f"Projection des individus sur 'F{i+1}' et 'F{i+2}'")

        # ajouter les axes
        axes[0].plot([projected_data[f'F{i+1}'].min(),projected_data[f'F{i+1}'].max()],[0,0],color='silver',linestyle='--',linewidth=3)
        axes[0].plot([0,0],[projected_data[f'F{i+2}'].min(),projected_data[f'F{i+2}'].max()],color='silver',linestyle='--',linewidth=3)

        # --------------- cercle des corrélations ---------------
        axes[1].set_xlim(-1,1)
        axes[1].set_ylim(-1,1)

        # affichage des étiquettes (noms des variables)
        for j in range(p):
            axes[1].annotate(data.columns[j],(corvar[j,i],corvar[j,i+1]))
            axes[1].arrow(0, 0, corvar[j,i], corvar[j,i+1], length_includes_head=True, head_width=0.04)

        # ajouter un cercle
        cercle = plt.Circle((0,0),1,color='blue',fill=False)
        axes[1].add_artist(cercle)

        # ajouter les axes
        axes[1].plot([-1,1],[0,0],color='silver',linestyle='--',linewidth=3)
        axes[1].plot([0,0],[-1,1],color='silver',linestyle='--',linewidth=3)

        # nom des axes, avec le pourcentage d'inertie expliqué
        axes[1].set_xlabel('F{} ({}%)'.format(i+1, round(100*acp.explained_variance_ratio_[i],1)))
        axes[1].set_ylabel('F{} ({}%)'.format(i+2, round(100*acp.explained_variance_ratio_[i+1],1)))
        axes[1].set_title(f"Cercle des corrélations (F{i+1} et F{i+2})")

        # affichage
        plt.show()