import tensorflow as tf

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