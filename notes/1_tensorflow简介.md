# Tensorflow 基本概念
-  使用**图**(Graphs)来表示计算任务
-  在被称之为**会话**(session)的**上下文**(context)中执行图
-  使用**tensor**表示数据
-  使用**变量**(Variable)维护状态
-  使用**feed**和**fetch**可以作为任意的操作复制或者从中获取数据

## 图(Graph)
---

```
# 创建图、并启动图

import tensorflow as tf

m1 = tf.constant([[3, 3]])          # 创建一个常量op
m2 = tf.constant([[2], [3]])        # 创建一个常量op
product = tf.matmul(m1, m2)         # 创建一个矩阵乘法op，把m1m2传入
print(product)                      # 此时不能正常打印出矩阵相乘的结果

sess = tf.Session()                 # 定一个会话，启动默认图
result = sess.run(product)          # 调用sess的润方法执行矩阵乘法op，run(product)触发三个op

print(result)
sess.close()

```

## 变量(Variable)
-  定义方法： `tf.Variable()`
-  变量初始化：`tf.global_variables_initializer()`
---
### 运算符: 
表达式 | 意思
:--:|:--:|
`tensorflow.add(a, b)`         | a + b
`tensorflow.subtract(a, b)`    | a - b
### 举例：

```
import tensorflow as tf

x = tf.Variable([1, 2])
a = tf.constant([3, 3])

# 增加一个减法op
sub = tf.subtract(x, a)
# 增加一个加法op
add = tf.add(x, sub)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

# 创建一个变量，初始化为0
state = tf.Variable(0, name = 'counter')
# 定义一个op，作用是使state加1
new_value = tf.add(state, 1)
# 赋值op，后给前
update = tf.assign(state, new_value)
# 变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
```
输出结果：
```
[-2 -1]
[-1  1]
0
1
2
3
4
5
```
## Fetch & Feed
-  Fetch：一个会话执行多个op，得到运行的结果
-  Feed：Feed的数据以字典形式传入，类似于传参

举例：
---
```
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
    print(sess.run(output, feed_dict={input4:[7.], input5: [2.]}))

```
运行结果：
```
[21.0, 7.0]
[14.]
```