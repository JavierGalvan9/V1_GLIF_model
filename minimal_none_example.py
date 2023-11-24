# %%
import tensorflow as tf

# a tensor that mimicks spiking neurons.

spikes = tf.ones([100, 10])


# pretend to have 3 networks that computes something.
# all 100 x 100 identical matrix is fine for now.


# 1: make sparse tensor from a trainable variable and
n1_values = tf.Variable(tf.ones([100]), trainable=True)
network1 = tf.sparse.SparseTensor(indices=[[i, i] for i in range(100)], values=n1_values, dense_shape=[100, 100])

# 2: use a sparse tensor constructed from a dense matrix
n2_mat = tf.Variable(tf.eye(100), trainable=True)
network2 = tf.sparse.from_dense(n2_mat)

# 3: normal dense matrix
network3 = tf.Variable(tf.eye(100), trainable=True)

# 4: dense matrix, but converted to sparse tensor right before use
network4 = tf.Variable(tf.eye(100), trainable=True)

# calculate some metrics from the spiking tensor
@tf.function
def get_loss(spikes, network1, network2, network3, network4):
    v1 = tf.sparse.sparse_dense_matmul(network1, spikes)
    v2 = tf.sparse.sparse_dense_matmul(network2, spikes)
    v3 = tf.linalg.matmul(network3, spikes)
    v4 = tf.sparse.sparse_dense_matmul(tf.sparse.from_dense(network4), spikes)
    vsum = v1 + v2 + v3 + v4
    return tf.reduce_mean(vsum)


# calculate gradients
with tf.GradientTape() as tape:
    loss = get_loss(spikes, network1, network2, network3, network4)

grads = tape.gradient(loss, [n1_values, n2_mat, network3, network4])
print(grads) # grads for network3 and network4 are calculated, but None for n1_values and n2_mat
# exit()

# print the gradients
# %% what if things are not calculated in a function?


with tf.GradientTape() as tape:
    v1 = tf.sparse.sparse_dense_matmul(network1, spikes)
    v2 = tf.sparse.sparse_dense_matmul(network2, spikes)
    v3 = tf.linalg.matmul(network3, spikes)
    v4 = tf.sparse.sparse_dense_matmul(tf.sparse.from_dense(network4), spikes)
    vsum = v1 + v2 + v3 + v4
    loss = tf.reduce_mean(vsum)

grads = tape.gradient(loss, [n1_values, n2_mat, network3, network4])
grads # gradients for 3 and 4 are calculated, but still None for 1 and 2

# %% If you contain the sparse network generation into the gradient tape scope, it works.

with tf.GradientTape() as tape:
    network1 = tf.sparse.SparseTensor(indices=[[i, i] for i in range(100)], values=n1_values, dense_shape=[100, 100])
    v1 = tf.sparse.sparse_dense_matmul(network1, spikes)
    network2 = tf.sparse.from_dense(n2_mat)
    v2 = tf.sparse.sparse_dense_matmul(network2, spikes)
    v3 = tf.linalg.matmul(network3, spikes)
    v4 = tf.sparse.sparse_dense_matmul(tf.sparse.from_dense(network4), spikes)
    vsum = v1 + v2 + v3 + v4
    loss = tf.reduce_mean(vsum)

grads = tape.gradient(loss, [n1_values, n2_mat, network3, network4])
grads # all the gradients are calculated