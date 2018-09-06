#Hello everryone and welcome to an introduction to tensorflow synatx and basics!
#This tutorial is brought to you by @astronomer_amber <3
#     -Using Python 3.6 and Atom editor w/ hyrodrogen,
#      feel free to copy and paste into Jupyter notebooks
#First make sure that you have tensorflow installed

import tensorflow as tf
import numpy as np
print (tf.__version__)

#Let's try running a session
hmm = tf.constant("you are fabulous")
with tf.Session() as sess: #with runs our operations in block of code then closes Session
    result = sess.run(hmm)

print(result)  #b is a bytes literal

a = tf.constant(2)
b = tf.constant(5)

with tf.Session() as sess:
    result = sess.run(a*b)

print(result)

#Woohoo you're doing awesome, now let's play with some matrices
matrixA = tf.constant([ [2], [3]])
matrixA.get_shape()
matrixB = tf.constant([ [10,1],[1,10]])
matrixB
matrixB.get_shape()
matrixC = tf.fill([2,2],7) #2x2 matrix filled with 7's

with tf.Session() as sess:
    result1 = tf.matmul(matrixB, matrixA)
    x = sess.run(result1)
    result2 = tf.matmul(matrixC, matrixA)
    y = sess.run(result2)
print(x)

sess = tf.InteractiveSession() #Allows me to run session in between cells
                                #only useful for Jupyter notebooks at hyrodrogen
result1 = tf.matmul(matrixB, matrixA)
sess.run(result1)
result1.eval() #another way to view results

#Graphs
#    -sets of nodes(vertices) connected by 'edges'
#    -In TF these nodes represent operations
#    -Tensor objects include: variables(must initalize) & placeholders
#Let's construct and execute!

node1 = tf.constant(10) #input variable1
node2 = tf.constant(-2) #input variable2
node3 = node1 + node2 #operation

with tf.Session() as sess:
    G1 = sess.run(node3)

print(G1)

VariableA = tf.Variable(10)
TensorA = tf.random_uniform((2,2),0,1) #random values from a uniform distribution
VariableB = tf.Variable(initial_value=TensorA)

init = tf.global_variables_initializer()
#run initalization
with tf.Session() as sess:
    G2 = sess.run(init)
    VarB = sess.run(VariableB)

print(VarB)
PlaceholderA = tf.placeholder(tf.float32) #specify type

np.random.seed(101)
tf.set_random_seed(101)

data1 = np.random.uniform(0,100,(3,3))
data2 = np.random.uniform(0,100,(3,1))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

with tf.Session() as sess:
    add_a_b = sess.run(a+b,feed_dict={a:5,b:10}) #can also multiply
    add_data1_data2 = sess.run(a+b,feed_dict={a:data1,b:data2})

print(add_a_b)
print(add_data1_data2)
