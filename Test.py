import tensorflow as tf
import pandas as pd

originalDataset = pd.read_csv('data.csv')
dataset = originalDataset.drop('Unnamed: 32', axis=1)
dataset['diagnosis'] = dataset['diagnosis'].map({'B':0, 'M':1})
datasetCorr = dataset.corr()

print("Corelations:- \n")
print(datasetCorr['diagnosis'].sort_values(ascending=False))
print("\n")

selectedFeatures = ['concave points_worst', 'perimeter_worst', 
                    'concave points_mean', 'perimeter_worst', 
                    'concave points_mean', 'radius_worst',
                    'perimeter_mean', 'area_worst', 'radius_mean', 
                    'area_mean', 'concavity_mean', 'concavity_worst', 
                    'compactness_mean', 'compactness_worst']
dataset = dataset[selectedFeatures + ['diagnosis']]





























              

