import tensorflow as tf
import numpy as np

# 使用numpy生成100个随机点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

# 构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
# y_data - y -> 真实值减去测试值 -> 误差
# tf.square(y_data - y) -> 误差的平方
# tf.reduce_mean(tf.square(y_data - y)) 误差的平方的平均值

# 定义一个梯度下降法来进行训练的优化器, 括号里是学习率
optimizer = tf.train.GradientDescentOptimizer(0.2)

# 最小化代价函数，最小化loss，越小越接近真实值
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# session
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run([k, b]))