import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder('float',shape=[None,784])
y_ = tf.placeholder('float',shape=[None,10])

# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))

def weight_variable(shape,name=""):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=name)


def bias_variable(shape,name=""):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


W_conv1 = weight_variable([5,5,1,32],name="W_conv1")
b_conv1 = bias_variable([32],name="b_conv1")

x_image = tf.reshape(x,[-1,28,28,1])

# 第一层卷积
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 卷积结果


# 第二次卷积
W_conv2 = weight_variable([5,5,32,64],name="W_conv2")
b_conv2 = bias_variable([64],name="b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
W_fc1 = weight_variable([7*7*64,1024],name="W_fc1")
b_fc1 = bias_variable([1024],name="b_fc1")

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

# Dropout 过拟合
keep_prob = tf.placeholder('float')   # 过拟合率
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,10],name="W_fc2")
b_fc2 = bias_variable([10],name="b_fc2")

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# 训练和评估
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))  # 计算交叉熵
# train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)  # 学习算法
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))  #x0= argmax(f(x)) 的意思就是参数x0满足f(x0)为f(x)的最大值；换句话说就是 argmax(f(x))是使得 f(x)取得最大值所对应的变量x。arg即argument，此处意为“自变量”
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
#cast(x, dtype, name=None)
# 将x的数据格式转化成dtype.例如，原来x的数据格式是bool，
# 那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()  #保存模型
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy = accuracy.eval(session=sess,feed_dict={
            x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d, training accuracy %g"%(i,train_accuracy))
    sess.run(train_step,feed_dict={x:batch[0],y_: batch[1], keep_prob : 0.5})

save_path = saver.save(sess,"Save/New_demo/first.ckpt")  # 保存模型

print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
    x: mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))


