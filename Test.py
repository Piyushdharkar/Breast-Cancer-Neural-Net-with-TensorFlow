import tensorflow as tf
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

originalDataset = pd.read_csv('data.csv')
dataset = originalDataset.drop('Unnamed: 32', axis=1)
dataset['diagnosis'] = dataset['diagnosis'].map({'B':0, 'M':1})
datasetCorr = dataset.corr()

#print("Corelations:- \n")
#print(datasetCorr['diagnosis'].sort_values(ascending=False))
#print("\n")

selectedFeatures = ['concave points_worst', 'perimeter_worst', 
                    'concave points_mean', 'perimeter_worst', 
                    'concave points_mean', 'radius_worst',
                    'perimeter_mean', 'area_worst', 'radius_mean', 
                    'area_mean', 'concavity_mean', 'concavity_worst', 
                    'compactness_mean', 'compactness_worst']
dataset = dataset[selectedFeatures + ['diagnosis']]


#Train Test Split
xtrain, xtest, ytrain, ytest = train_test_split(dataset.drop('diagnosis', axis=1),
                                dataset['diagnosis'])


features = 14
first_hidden_layer_size = 8
output_size = 1
learning_rate = 0.8
epochs = 1000

#Input Layer
x = tf.placeholder(tf.float32, [None, features])

#First Hidden Layer
W1 = tf.Variable(tf.random_normal([features, first_hidden_layer_size]))
b1 = tf.Variable(tf.random_normal([first_hidden_layer_size]))
z1 = tf.matmul(x, W1) + b1
y1 = tf.nn.softmax(z1)

#Output Layer
W2 = tf.Variable(tf.random_normal([first_hidden_layer_size, output_size]))
b2 = tf.Variable(tf.random_normal([output_size]))
z2 = tf.matmul(y1, W2) + b2
y2 = tf.nn.softmax(z2)

#Labels
y_ = tf.placeholder(tf.float32, [None, 1])

#Cost function
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y2, labels=output))
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y2), [1]))

#Training
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

X, Y = xtrain, pd.DataFrame(ytrain)
for _ in range(epochs):
    sess.run(train_step, feed_dict={x:X, y_:Y})


correct_prediction = tf.equal(y2, y_)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

one, two, three = sess.run([accuracy, y_, y2], feed_dict={x:xtest, y_:pd.DataFrame(ytest)})

print(one)

print(two)

print(three)





















              

