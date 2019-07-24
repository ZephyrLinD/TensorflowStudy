import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
minst = input_data.read_data_sets("Datasets/MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100

# 计算一共有多少个批次(//为整除)
n_batch = minst.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])     # 每个数字 28 * 28 = 784,把图片拉成向量
y = tf.placeholder(tf.float32, [None, 10])      # 一共0-9，10个数字

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)  # matmul为矩阵相乘。x * W + b表示信号的综合，在经过一个softmax函数，将输出的信号转化为概率值

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))

# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
corrent_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# equal：比较大小是否一样，一样true不一样false
# argmax：求最大值在哪个位置，返回最大值的位置(返回一维张量中最大值所在的位置)

# 求准确率
accurancy = tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = minst.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
        
        acc = sess.run(accurancy, feed_dict={x:minst.test.images, y:minst.test.labels})
        print("Iter " + str(epoch) + ", Testing Accurency " + str(acc))