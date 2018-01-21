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



features = 13
first_hidden_layer_size = 4
output_size = 5

#Input Layer
x = tf.placeholder(tf.float32, [None, features])

#First Hidden Layer
W1 = tf.Variable(tf.random_normal([features, first_hidden_layer_size]))
b1 = tf.Variable(tf.random_normal([first_hidden_layer_size]))
z1 = tf.matmul(x, W1) + b1
y1 = tf.nn.relu(z1)

#Output Layer
W2 = tf.Variable(tf.random_normal([first_hidden_layer_size, output_size]))
b2 = tf.Variable(tf.random_normal([output_size]))
z2 = tf.matmul(y1, W2) + b2
y2 = tf.nn.relu(z2)

































              

