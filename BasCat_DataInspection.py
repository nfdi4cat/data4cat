# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to approach remote data - A NFDI4Cat use case.

# %% [markdown]
# ## Introduction
#
# Working with partners in a joined project can very often have challenges. This is especially true when work packages have interfaces to other ones. In [NFDI4Cat](https://doi.org/10.1002/cctc.202001974) the challenge is to work data across a consortium. To enhance this processs especially also for teaching purposes data provided by partners was wrapped into a package in order to facilitate its usage and to get started with the process of data analytic.
#
# ## Basic data inspection on the BasCat provided perfomance data set

# %%
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ### The tools used
#
# In this work several tools will be used. The most important thing is Python which is used for all computations. As an interface to the Python interpreter JupyterLab is used which gives a modern user interface within the browser. In order to create a plain text format out of the Jupyter notebook Jupytext is used in order to reduce the less relevant meta information in the notebook. Special libraries used will be discussed when they are used in the data analysis.
#
# ### Importing the dataset from the Dataverse instance
#
# The data storage pilot instance for NFDI4Cat and in the future probably the whole catalysis community in Germany is installed with help of Fraunhofer Fokus as a [Dataverse instance](https://dataverse.org/). To get an initial glimpse on the data in the repository we created a helper package that provides the dataset in a scikit-learn like fashion. For further instructions also compare the Readme file in the data4cat repository.

# %% [markdown]
# ## Download and basic inspection
#
# In order to first download the dataset we use the data4cat package. The requests calls used to communicate with the Dataverse REST API are hidden in this package for convenience.

# %%
from data4cat import dino_run

# %%
dinodat = dino_run.dino_offline()

# %% [markdown]
# The next is a representation of the data available online.

# %%
original_data = dinodat.original_data()
original_data.head()

# %% [markdown]
# The first experiment, which runs until ~85h can be used to observe stabilization behavior of catalysts, in terms of CO conversion (X_CO). This can be seen in the following visualization.

# %%
startup_data = dinodat.startup_data()
startup_data.tail()

# %%
fig = plt.figure(figsize=(12,6))
for entry in [1,2,3,4]:
    
    filtered_by_reactor = startup_data[startup_data['Reactor']== entry]  
    
    ax1 = fig.add_subplot(2,2,entry)
    ax1.set_title(r'$X_{CO}$ vs TOS')
    ax1.set_xlabel('Time on stream [h]')
    ax1.set_ylabel('CO conversion []')
    left, = ax1.plot(filtered_by_reactor['TOS'], filtered_by_reactor['X_CO'],'bo-')
    
    ax2 = ax1.twinx()
    right, = ax2.plot(filtered_by_reactor['TOS'], np.gradient(filtered_by_reactor['X_CO']),'rd-')
    ax2.set_ylabel('Gradient')
    ax1.legend([left, right], ['$X_{CO}$ Reactor '+ str(entry),'Gradient'], bbox_to_anchor=(0.65, 0.65), loc=2, borderaxespad=0.)
    fig.tight_layout();

# %% [markdown]
# For further analysis unsupervised and supervised learning will be used. We therefore need different representations of the data. The selectivity matrix will be used as features (x-values). The reactor numbering will be used as initial "clusters" (y-values).

# %%
X, clusters = dinodat.selectivity()
X.head()

# %% [markdown]
# ## Unsupervised Learning
#
# First we do some dimensional reduction for the 25 dimensional feature matrix X. For this the PCA and the t-SNE algorithm are used. A random seed is used to enhance deterministic results.

# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm

np.random.seed(42)

# %%
pca = PCA(n_components=2).fit_transform(X)
tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(X)

# %%
title = 'Catalyst'
labels = [r'$Rh/SiO_2$',r'$RhMn/SiO_2$',r'$RhFe/SiO_2$',r'$RhMnFe/SiO_2$',r'$empty$']

fig, ax = plt.subplots(figsize=(7,4))
plt.title('Principal component analysis')
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
scatterplot = ax.scatter(pca[:,0], pca[:,1], c=clusters, cmap=cm.Set1)
ax.legend(handles=scatterplot.legend_elements()[0],labels=labels, title=title, bbox_to_anchor=(1.5, 1.0))
plt.tight_layout();

# %%
title = 'Catalyst'
labels = [r'$Rh/SiO_2$',r'$RhMn/SiO_2$',r'$RhFe/SiO_2$',r'$RhMnFe/SiO_2$',r'$empty$']

fig, ax = plt.subplots(figsize=(7,4))
plt.title('T-distributed stochastic neighbor embedding')
plt.xlabel('TSNE component 1')
plt.ylabel('TSNE component 2')
scatterplot = ax.scatter(tsne[:,0], tsne[:,1], c=clusters, cmap=cm.Set1)
ax.legend(handles=scatterplot.legend_elements()[0],labels=labels, title=title, bbox_to_anchor=(1.5, 1.0))
plt.tight_layout();

# %% [markdown]
# The overall initial inspection of the data show that we should remove the empty reactor from the dataset. The dimensional reduction shows 5 clusters which can be identified by the reactor tubes.

# %%
title = 'Catalyst'
labels = [r'$Rh/SiO_2$',r'$RhMn/SiO_2$',r'$RhFe/SiO_2$',r'$RhMnFe/SiO_2$',r'$empty$']

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
fig.suptitle('Unsupervised Learning on Dino Run-46 Performance')

ax1.set_title('PCA')
ax1.set_xlabel('PCA component 1')
ax1.set_ylabel('PCA component 2')
scatterplot1 = ax1.scatter(pca[:,0], pca[:,1], c=clusters, cmap=cm.Set1)

ax2.set_title('T-SNE')
ax2.set_xlabel('TSNE component 1')
ax2.set_ylabel('TSNE component 2')
scatterplot2 = ax2.scatter(tsne[:,0], tsne[:,1], c=clusters, cmap=cm.Set1)

ax2.legend(handles=scatterplot2.legend_elements()[0],labels=labels, title=title, bbox_to_anchor=(1.5, 1.0))

plt.tight_layout();

# %% [markdown]
# We therefore have to filter the empty reactor from the dataset. The data4cat package as an included argument for this. We now create a new feature matrix and "clusters".

# %%
X_filtered, clusters = dinodat.selectivity(r5=False)
X_filtered.head()

# %% [markdown]
# Next we do a new PCA and t-SNE with the reduced dataset.

# %%
pca = PCA(n_components=2).fit_transform(X_filtered)
tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(X_filtered)

# %% [markdown]
# Again the dataset can be distiguished by the reactor/catalyst.

# %%
title = 'Catalyst'
labels = [r'$Rh/SiO_2$',r'$RhMn/SiO_2$',r'$RhFe/SiO_2$',r'$RhMnFe/SiO_2$']

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
fig.suptitle('Unsupervised Learning on Dino Run-46 Performance')

ax1.set_title('PCA')
ax1.set_xlabel('PCA component 1')
ax1.set_ylabel('PCA component 2')
scatterplot1 = ax1.scatter(pca[:,0], pca[:,1], c=clusters, cmap=cm.Set1)

ax2.set_title('T-SNE')
ax2.set_xlabel('TSNE component 1')
ax2.set_ylabel('TSNE component 2')
scatterplot2 = ax2.scatter(tsne[:,0], tsne[:,1], c=clusters, cmap=cm.Set1)

ax2.legend(handles=scatterplot2.legend_elements()[0],labels=labels, title=title, bbox_to_anchor=(1.5, 1.0))

plt.tight_layout()

# %% [markdown]
# For further insights the reactor tube is replaced by that catalyst composition. This also exists in data4cat for convenience.

# %%
X, y = dinodat.react_cond(r5=False)

# %% [markdown]
# This leads to a reduced dataset with all the setpoints as features and the selectivity to ethanol as target.

# %%
X.head()

# %% [markdown]
# Now we can also cluster by the Rh content. It seems like the catalyst composition has definitely some impact on the data.

# %%
title = 'Catalyst'
labels = ['Rh = 2.12', 'Rh = 2.52', 'Rh = 2.46']

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
fig.suptitle('Unsupervised Learning on Dino Run-46 Performance')

ax1.set_title('PCA')
ax1.set_xlabel('PCA component 1')
ax1.set_ylabel('PCA component 2')
scatterplot1 = ax1.scatter(pca[:,0], pca[:,1], c=X['Rh'], cmap=cm.Set1)

ax2.set_title('T-SNE')
ax2.set_xlabel('TSNE component 1')
ax2.set_ylabel('TSNE component 2')
scatterplot2 = ax2.scatter(tsne[:,0], tsne[:,1], c=X['Rh'], cmap=cm.Set1)

ax2.legend(handles=scatterplot2.legend_elements()[0],labels=labels, title=title, bbox_to_anchor=(1.5, 1.0))

plt.tight_layout();

# %% [markdown]
# The initial clustering will now be complemented with clustering that does not now anything about the reactor setup. We use the Kmeans algorithm for this purpose.

# %%
from sklearn.cluster import KMeans

# %%
k_ellbow = []
for E in range(20):
    E = E+1
    kmeans = KMeans(n_clusters=E, n_init='auto')
    kmeans.fit(X_filtered)
    score = kmeans.score(X_filtered)
    k_ellbow.append(score)

# %%
plt.plot(np.gradient(k_ellbow))
plt.xlabel('n Cluster')
plt.ylabel('grad Score');

# %%
k_cluster = KMeans(n_clusters=4, n_init='auto')
k_cluster.fit(X_filtered)
y_k_cluster = k_cluster.predict(X_filtered)

# %% [markdown]
# Also when doing a completely unsupervised clustering the same clusters can be identified which are the same as the reactors/catalysts.

# %%
title = 'Catalyst'
labels = [r'$Rh/SiO_2$',r'$RhMn/SiO_2$',r'$RhFe/SiO_2$',r'$RhMnFe/SiO_2$']
labels1 = ['Rh = 2.12', 'Rh = 2.52', 'Rh = 2.46']
labels2 = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
fig.suptitle('Unsupervised Learning on Dino Run-46 Performance')

ax1.set_title('T-SNE')
ax1.set_xlabel('TSNE component 1')
ax1.set_ylabel('TSNE component 2')
scatterplot1 = ax1.scatter(tsne[:,0], tsne[:,1], c=clusters, cmap=cm.Set1)

ax2.set_title('T-SNE')
ax2.set_xlabel('TSNE component 1')
ax2.set_ylabel('TSNE component 2')
scatterplot2 = ax2.scatter(tsne[:,0], tsne[:,1], c=y_k_cluster, cmap=cm.Set1)

ax1.legend(handles=scatterplot1.legend_elements()[0],labels=labels, title=title, bbox_to_anchor=(1.5, 1.0))
ax2.legend(handles=scatterplot2.legend_elements()[0],labels=labels2, title=title, bbox_to_anchor=(1.5, 1.0))

plt.tight_layout();

# %% [markdown]
# ## Supervised learning

# %% [markdown]
# Next supervised learning will be used to find more dependencies within the data. We choose a Random Forest Regressor to watch out for those dependencies.

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# %%
X.head()

# %%
y.head()

# %%
names = X.columns

# %%
forest = RandomForestRegressor()
forest.fit(X, y)

# %%
importances = forest.feature_importances_

# %% [markdown]
# Looking at the feature importance clearly shows that the selectivity is governed by the Rh content and a little bit by the TOS. The rest of the features do not play a big role.

# %%
plt.xticks(rotation = 90)
plt.bar(x=names, height=importances);

# %% [markdown]
# This can also be expressed via the next two pictures.

# %%
plt.xlabel('Rh')
plt.ylabel('S_Ethanol')
plt.scatter(X['Rh'], y, c=X['TOS'])
plt.colorbar(label='TOS');

# %%
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.xlabel('Rh')
plt.ylabel('S_Ethanol')
plt.scatter(X['Rh'], y, c=X['TOS'])
plt.colorbar(label='TOS')

plt.subplot(122)
plt.xlabel('TOS')
plt.ylabel('S_Ethanol')
plt.scatter(X['TOS'], y, c=X['Rh'])
plt.colorbar(label='Rh')
plt.tight_layout();

# %% [markdown]
# We can use this information to normalize the ethanol selectivity by the TOS and the Rh content. This leads to the following histogram where we end up with basically the same number for all experiments.

# %%
plt.xlabel('normed Selectivity')
plt.ylabel('Value count')
plt.hist(y/(X['TOS']*X['Rh']));

# %%
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
fig.suptitle('Supervised Learning on Dino Run-46 Performance')

ax1.set_title('Feature importance')
ax1.set_ylabel('Importance')
ax1.bar(x=names, height=importances)
ax1.set_xticks(ax1.get_xticks(), ax1.get_xticklabels(), rotation=45, ha='right')

ax2.set_title('Normalized selectivity')
ax2.set_xlabel('Value')
ax2.set_ylabel('Value count')
ax2.hist(y/(X['TOS']*X['Rh']))
ax2.set_xticks(ax2.get_xticks(), ax2.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout();

# %% [markdown]
# For reproducibility finally a requirements file is written.

# %%
import session_info
session_info.show(write_req_file=True, req_file_name='requirements.txt')
