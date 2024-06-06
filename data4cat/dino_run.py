#!/usr/bin/python3

import json, requests
import pandas as pd
from io import BytesIO
from sklearn.utils import Bunch

config = {'filepath': 'data4cat/', 'dataset': 'dino_dataset.json', 'csv_name': 'dino_run.csv'}

class dino_offline:

    def __init__(self):
        self.filepath = config['filepath']
        self.dataset = config['dataset']
        self.csv_name = config['csv_name']
        
        print('Instance created')

    def create_dictionary(self, filepath=None, dataset=None) -> None:
        
        if filepath==None and dataset==None:
            filepath, dataset = self.filepath, self.dataset
        
        dino_dataset ={
            'url' : 'https://nfdirepo.fokus.fraunhofer.de',
            'ID' : 'doi:10.82207/FK2/NR5BWO',
            'root_ID' : 'root',
            'dataset_ID' : '68',
            'version' : '2.0'
        } 
    
        with open(self.filepath + self.dataset, 'w') as outfile: 
            json.dump(dino_dataset, outfile)

        print('Dictionary created')
    
    def read_dictionary(self, filepath=None, dataset=None):
        
        if filepath==None and dataset==None:
            filepath, dataset = self.filepath, self.dataset
        
        with open(self.filepath + self.dataset, 'r') as readfile: 
            data = json.load(readfile)

        print('Dictionary read')
        
        return data
    
    def get_dataset(self, data_dict=None):

        if data_dict==None:
            data_dict = self.read_dictionary()
            
        create_url = data_dict['url']+'/api/access/datafile/'+data_dict['dataset_ID']+'?persistentId='+data_dict['ID']
        get_original = requests.get(create_url, params='format=original')
        xs_vsTOS = pd.read_excel(BytesIO(get_original.content), sheet_name=4, engine='openpyxl')
        
        print('Dataset downloaded')
        
        return xs_vsTOS   
    
    def create_csv(self, filepath=None, csv_name=None):
        
        if filepath==None and csv_name==None:
            filepath, dataset = self.filepath, self.csv_name

        data = self.get_dataset()
        data.to_csv(self.filepath + self.csv_name, index=False)

        print('Stored dataset')

    def one_shot_dumb(self):
        self.create_dictionary()
        self.create_csv()
        print('Done')

    def original_data(self, filepath=None, csv_name=None, output='p'):
        '''
        Inspired by
        https://stackoverflow.com/questions/53090837/how-to-transform-my-csv-file-into-this-scikit-learn-dataset?noredirect=1&lq=1
        '''
        if filepath==None and csv_name==None:
            filepath, dataset = self.filepath, self.csv_name

        pd_data = pd.read_csv(self.filepath + self.csv_name)
        feature_names = list(pd_data.columns)
        data = pd_data.to_numpy()

        if output == 'b':
            return Bunch(data=data, target=None, feature_names = feature_names, target_names = None)
        if output == 'p':
            return pd_data
    
    def startup_data(self, filepath=None, csv_name=None, output='p'):
        
        if filepath==None and csv_name==None:
            filepath, dataset = self.filepath, self.csv_name

        pd_data = pd.read_csv(self.filepath + self.csv_name)
        pd_data = pd_data[pd_data['TOS']< 85]
        X, y = pd_data[['TOS', 'X_CO']], pd_data['Reactor']
        feature_names, target_names = list(X.columns), [y.name]
        data, target = X.to_numpy(), y.to_numpy()
        
        if output == 'b':
            return Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)
        if output == 'p':
            return pd_data[['TOS', 'X_CO', 'Reactor']]

    def selectivity(self, filepath=None, csv_name=None, output='p', r5=True):
        
        if filepath==None and csv_name==None:
            filepath, dataset = self.filepath, self.csv_name

        pd_data = pd.read_csv(self.filepath + self.csv_name)
        
        if r5 == False:
            pd_data = pd_data[pd_data['Reactor']!=5]
            
        X, y = pd_data.iloc[:,11:-2], pd_data['Reactor']
        feature_names, target_names = list(X.columns), [y.name]
        data, target = X.to_numpy(), y.to_numpy()
        
        if output == 'b':
            return Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)
        if output == 'p':
            return pd_data.iloc[:,11:-2], pd_data['Reactor']

    def react_cond(self, filepath=None, csv_name=None, output='p', r5=True):

        cataDict = {
            1:'2.12,0,0',
            2:'2.52,1.54,0',
            3:'2.52,0,1.48',
            4:'2.46,1.46,1.46',
            5:'0,0,0'
        }
        
        if filepath==None and csv_name==None:
            filepath, dataset = self.filepath, self.csv_name

        pd_data = pd.read_csv(self.filepath + self.csv_name)
        
        if r5 == False:
            pd_data = pd_data[pd_data['Reactor']!=5]
            selectivity_EtOH = pd_data['S_Ethanol']
        else:
            selectivity_EtOH = pd_data['S_Ethanol']

        pd_data = pd_data.iloc[:,2:11]
        pd_data = pd_data.replace({'Reactor':cataDict})

        pd_data[['Rh', 'Mn', 'Fe']] = pd_data['Reactor'].str.split(pat=',', expand=True)
        pd_data = pd_data.drop(['Reactor'], axis=1)
        
        pd_data[['Rh', 'Mn', 'Fe']] = pd_data[['Rh', 'Mn', 'Fe']].astype('float')
        
        X, y = pd_data, selectivity_EtOH
        feature_names, target_names = list(X.columns), [y.name]
        data, target = X.to_numpy(), y.to_numpy()
        
        if output == 'b':
            return Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)
        if output == 'p':
            return pd_data, selectivity_EtOH

if __name__ == "__main__":
    print('Main programme for dino_run')


