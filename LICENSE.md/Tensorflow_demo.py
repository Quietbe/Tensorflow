import tensorflow as tf
from PIL import Image
import numpy as np




def loadImage(path):
    im = Image.open(path)
    im = im.resize((28,28),Image.BILINEAR)  # 缩放图片为28X28
    im = im.convert("L")    # 将图像转换成灰度图像
    data = im.getdata()      # 得到数组对象
    data = np.matrix(data)   # 数据转换为np.matrix对象
    data = np.reshape(data, (1, 784))   # 格式转换
    data = data - data.min()
    return data

sess = tf.Session()

x = tf.placeholder('float',shape=[None,784])
# x = loadImage('0.bmp')
# y = tf.placeholder('float',shape=[None,10])


# 原来算法还原

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

saver = tf.train.Saver()
saver.restore(sess,"Save/New_demo/first.ckpt")
result = sess.run(y_conv,feed_dict={x:loadImage('9.bmp'),keep_prob:1.0})

print("result:",result)
n = 0
for i in result[0]:
    print("为%s的概率:%s"%(n,round(i,3)))
    n = n+1

