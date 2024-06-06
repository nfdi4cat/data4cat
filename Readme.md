---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Usage of the data4cat module

For convenience and e.g. the usage in lectures the BasCat DinoRun dataset was wrapped into some convenience functions to enable a smooth start on how to work with published remote data.

First import the module:


```
git remote add origin https://github.com/nfdi4cat/data4cat.git
git branch -M main
git push -u origin main
````


```
git clone https://github.com/nfdi4cat/data4cat.git
```
cd into the directory an install data4cat

```
pip install .
```

```
python -m pip install git+https://github.com/nfdi4cat/data4cat.git@main
````


```python
from data4cat import dino_run
```

Create an instance:

```python
dinodat = dino_run.dino_offline()
```

The two steps above have to be done always.


## Download the dino_run dataset from the NFDI4Cat Dataverse instance


In case that there is no offline version of the dataset available a copy of the dataset can be downloaded like this:

```python
download = 'n'
if download == 'y':
    dinodat.one_shot_dumb()
```

## Create a dataset from the offline data


You can get the data either in the form of a pandas dataframe or as a Bunch object in the style of scikit-learn datasets. You can get the original data in the following way:

```python
original = dinodat.original_data()
```

```python
original.head()
```

## Create a subset of the offline data for the startup phase


There is a sub dataset for the startup phase with a TOS < 85 available. Again both as pandas dataframe and Bunch object.

```python
startup = dinodat.startup_data()
```

```python
startup.head()
```

## Create a subset of the offline data for the selectivity


Especially for unsupervised learning tasks there is a subset of the data prepared that contains only the selectivity data. When asking for this subset also reactors are provided, here they are put in a clusters object.

```python
selectivity, clusters = dinodat.selectivity()
```

```python
selectivity.head()
```

```python
clusters.head()
```

## Create a subset of the offline data for the selectivity without reactor 5


I case needed when you provide the r5 argument to False it will exclude the empty reactor 5.

```python
selectivity_wo5, clusters = dinodat.selectivity(r5=False)
```

```python
selectivity_wo5.head()
```

```python
clusters.head()
```

## Create a subset of the offline data for the reaction conditions


For supervised tasks a subset of the data is provided that contains the reaction conditions as features and the selectivity to ethanol as target.

```python
react_cond, selectivity_EtOH = dinodat.react_cond()
```

```python
react_cond.head()
```

```python
selectivity_EtOH.head()
```

## Create a subset of the offline data for the reaction conditions without reactor 5


Like before the empty reactor 5 can be excluded with the r5 argument set to False.

```python
react_cond, selectivity_EtOH = dinodat.react_cond(r5=False)
```

```python
react_cond.tail()
```

```python
selectivity_EtOH.tail()
```
