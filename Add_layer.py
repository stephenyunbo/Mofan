import tensorflow as tf
import numpy as np

# 定义添加神经层的函数 def add_layer(), has four parameters: 输入值，输入的大小，输出的大小，激励函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size])) # Weights是一个in_size行, out_size列的随机变量矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1) # biases的推荐值不为0， 所以在0向量的基础上+0.1
    Wx_plus_b = tf.matmul(inputs, Weights) + biases # 定义神经网络未激活的值, tf.matmul refers to the 'matrix multiplication'
    """ When activation_function is None, the outputs is the current prediction, i.e., Wx_plus_b.  
        Otherwise the outputs is activation_function(Wx_plus_b)
    """
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    
    return outputs

# 构建所需要的数据
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32) # 使数据看起来更像真实情况, 否则就是一个标准的抛物线 
y_data = np.square(x_data) - 0.5 + noise


# 使用占位符定义我们所需的神经网络的输入。None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

"""
定义神经层，输入层只有一个属性，即只有一个输入，输出层的结构和输入层一样，隐藏层假设有10个神经元。
所以我们构建的是——输入层1个，隐藏层10个，输出层1个的神经网络
"""

# 定义隐藏层 输入只有1层，输出有10层; tf.nn.relu是tensorflow自带的激励函数
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# 定义输出层 这里的输入就是上面隐藏层的输出，即这里输入有10层，输出有1层
prediction = add_layer(l1, 10, 1, activation_function=None)

# 计算预测值与真实值的误差，对二者差的平方求和再取平均
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# tf.train.GradientDescentOptimizer()中的值通常都小于1，这里取0.1， 即以0.1的效率来最小化误差loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 对变量初始化
init = tf.global_variables_initializer()

# define Session
sess = tf.Session()
sess.run(init)

# Train 让机器学习1000次，学习的内容是train_step, 用Session来run每一次training的数据（当运算要用到placeholder时， 就需要feed_dict这个字典来指定输入）
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # 每50步输出一下机器学习的误差
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
