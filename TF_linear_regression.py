import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg'

learning_rate = 0.01
epochs = 200
n_samples = 30
X_train = np.linspace(0, 20, n_samples)
y_train = 2*X_train + np.random.randn(n_samples)

plt.figure(1,figsize = (6,4))
plt.ylabel('y_train', fontsize = 20)
plt.xlabel('X_train',fontsize = 20)

plt.scatter(X_train, y_train, marker ='*', c = 'purple', s=13, label = 'training data')
plt.legend(loc='upper left', prop={'size':10},frameon=False)
leg = plt.gca().get_legend()
leg.legendHandles[0].set_visible(False)
plt.tick_params(labelsize=20)

plt.show()

#Neural network time
n_features = 10
n_neurons = 3

x = tf.placeholder(tf.float32, (None,n_features)) #(samples, n_features)
W = tf.Variable(tf.random_normal([n_features,n_neurons]), name = 'weights')
b = tf.ones([n_neurons], name = 'bias')

z = tf.add(tf.matmul(x,W),b)
a = tf.sigmoid(z)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    layer = sess.run(a, feed_dict={x:np.random.random([1,n_features])})

print(layer)

#need to define/use cost function to adjust the weights(W) and bias(b) AKA backpropagation

X_data = np.linspace(0,10,10) - np.random.uniform(-2.0,2.0,10)
#or X_data = tf.placeholder(tf.float32)
y_data = np.linspace(0,10,10) - np.random.uniform(-2.0,2.0,10)
#or y_data = tf.placeholder(tf.float32)

#y= mx +b, y_pred = WX + b
m = tf.Variable(np.random.randn(), name = 'weights')
b = tf.Variable(np.random.randn(), name = 'bias')

cost = 0

for x,y in zip(X_data, y_data):
    y_predicted = m*x +b
    cost += (y - y_predicted)**2 #cost function

#can also write cost funtion as:
#y_predicted = m*x +b
#cost = tf.reduced_sum((y_data - y_predicted)**2)/ (2*n_features))

#optimize!
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) #initalize variables
    training = 100 #how many training steps to actually perform

    for i in range(training):
        sess.run(optimizer)

    m_model, b_model = sess.run([m,b])

plt.plot(X_data, y_data, 'go')
#y=mx+b
y = m_model*X_data + b_model
plt.plot(X_data, y, 'r-')
plt.show()
