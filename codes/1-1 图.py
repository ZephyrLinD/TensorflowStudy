# 创建图，启动图
import tensorflow as tf
m1 = tf.constant([[3, 3]])          # 创建一个常量op
m2 = tf.constant([[2], [3]])        # 创建一个常量op
product = tf.matmul(m1, m2)         # 创建一个矩阵乘法op，把m1m2传入
print(product)

sess = tf.Session()                 # 定一个会话，启动默认图
result = sess.run(product)          # 调用sess的润方法执行矩阵乘法op，run(product)触发三个op
print(result)
sess.close()
