import numpy as np
import tensorflow as tf
import time

a = time.time()
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*5+8*x_data*x_data+6

Weights = tf.Variable(tf.random_uniform([1],-1.0,10))
Weights2 = tf.Variable(tf.random_uniform([1],-1.0,10))
biases = tf.Variable(tf.zeros([1]))


y = Weights*x_data + Weights2*x_data*x_data+ biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))  # 调用GPU
# sess = tf.Session()
sess.run(init)

for step in range(4000):
    sess.run(train)
    if step %20 ==0:
        print(step,sess.run(Weights),sess.run(Weights2),sess.run(biases))

b = time.time()
print(b-a)