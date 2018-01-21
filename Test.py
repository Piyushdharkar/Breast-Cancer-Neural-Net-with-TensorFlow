import tensorflow as tf
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
import numpy as np

originalDataset = pd.read_csv('data.csv')
dataset = originalDataset.drop('Unnamed: 32', axis=1)
dataset['diagnosis'] = dataset['diagnosis'].map({'B':0, 'M':1})
datasetCorr = dataset.corr()

print("Corelations:- \n")
print(datasetCorr['diagnosis'].sort_values(ascending=False))
print("\n")

#Selected 14 most relevant features
selectedFeatures = ['concave points_worst', 'perimeter_worst', 
                    'concave points_mean', 'perimeter_worst', 
                    'concave points_mean', 'radius_worst',
                    'perimeter_mean', 'area_worst', 'radius_mean', 
                    'area_mean', 'concavity_mean', 'concavity_worst', 
                    'compactness_mean', 'compactness_worst']
dataset = dataset[selectedFeatures + ['diagnosis']]

dataset[selectedFeatures] = normalize(dataset[selectedFeatures])


#Train Test Split
xtrain, xtest, ytraint, ytestt = train_test_split(dataset.drop('diagnosis', axis=1),
                                dataset['diagnosis'])

#Conversion to one hot vector
ytrain =  np.zeros((len(ytraint), 2))
ytest = np.zeros((len(ytestt), 2))

ytrain[np.arange(len(ytraint)), ytraint] = 1
ytest[np.arange(len(ytestt)), ytestt] = 1

#Parameters of neural network
features = len(selectedFeatures)
output_size = 2
first_hidden_layer_size = int((len(selectedFeatures) + output_size) / 2)
learning_rate = 0.1
epochs = 1000

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
y2 = tf.nn.softmax(z2)

#Labels
y_ = tf.placeholder(tf.float32, [None, 2])

#Cost function
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y2), [1]))

#Training
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

X, Y = xtrain, pd.DataFrame(ytrain)
for _ in range(epochs):
    sess.run(train_step, feed_dict={x:X, y_:Y})

#Obtain accuracy score
correct_prediction = tf.equal(tf.arg_max(y2, 1), tf.arg_max(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

one, two, three = sess.run([accuracy, y_, y2], feed_dict={x:xtest, y_:pd.DataFrame(ytest)})

print(one)

print(two)

print(np.rint(three))





















              

