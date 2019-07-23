import tensorflow as tf

# Fetch: 一个会话执行多个op，得到运行的结果
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    result = sess.run([mul, add])
    print(result)


# Feed 
input4 = tf.placeholder(tf.float32)     # 占位符
input5 = tf.placeholder(tf.float32)     # 占位符
output = tf.multiply(input4, input5)

with tf.Session() as sess:
    # Feed的数据以字典形式传入
    print(sess.run(output, feed_dict={input4:[7.],input5: [2.]}))