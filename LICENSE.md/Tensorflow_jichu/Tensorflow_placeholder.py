import tensorflow as tf

# 传入值  placeholder 与 feed_dict 绑定 同时出现

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1,input2)   #mul  乘法运算

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.0]}))


