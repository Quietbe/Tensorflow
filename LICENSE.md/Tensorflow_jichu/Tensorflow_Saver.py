import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='Weights')
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='biases')
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 计算准确度
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.arg_max(y_pre,1),tf.arg_max(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# config=tf.ConfigProto(log_device_placement=True) 调用GPU

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()  #保存模型
for i in range(5000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 100 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
save_path = saver.save(sess,"Save/Save_demo01.ckpt")  # 保存模型

