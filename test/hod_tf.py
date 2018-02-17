import numpy as np
import tensorflow as tf

from sklearn.datasets import make_blobs

### Create Dataset

np.random.seed(666)
nb_samples = 2000
X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=2, cluster_std=1.1, random_state=2000)

# Transform the original dataset so to learn the bias as any other parameter
Xc = np.ones((nb_samples, X.shape[1] + 1), dtype=np.float32)
Yc = np.zeros((nb_samples, 1), dtype=np.float32)

Xc[:, 0:2] = X
Yc[:, 0] = Y

# Create Tensorflow graph
graph = tf.Graph()

with graph.as_default():
    Xi = tf.placeholder(tf.float32, Xc.shape)
    Yi = tf.placeholder(tf.float32, Yc.shape)

    # Weights (+ bias)
    W = tf.Variable(tf.random_normal([Xc.shape[1], 1], 0.0, 0.01))

    # Z = wx + b
    Z = tf.matmul(Xi, W)

    # Log-likelihood
    log_likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z, labels=Yi))

    # Cost function (Log-likelihood + L2 penalty)
    cost = log_likelihood + 0.5  # * tf.norm(W, ord=2)

    trainer = tf.train.GradientDescentOptimizer(0.0025)
    training_step = trainer.minimize(cost)

    # Compute the FIM
    dW = tf.gradients(-log_likelihood, W)
    FIM = tf.matmul(tf.reshape(dW, (Xc.shape[1], 1)), tf.reshape(dW, (Xc.shape[1], 1)), transpose_b=True)

    #    FIM_h = tf.gradients(tf.gradients(-log_likelihood, W)[0], W)[0]
    #    FIM_h = tf.hessians(-log_likelihood, tf.squeeze(W))
    # y_list = tf.unstack(y)
    hess_list = [tf.gradients(y_, W)[0] for y_ in tf.unstack(tf.squeeze(dW))]  # list [grad(y0, x), grad(y1, x), ...]
    FIM_h = tf.stack(hess_list)

# Create Tensorflow session
session = tf.InteractiveSession(graph=graph)

# Initialize all variables
tf.global_variables_initializer().run()

# Run a training cycle
# The model is quite simple, however a check on the cost function should be performed
for _ in range(3500):
    _, c, z_out = session.run([training_step, cost, Z], feed_dict={Xi: Xc, Yi: Yc})

# Compute Fisher Information Matrix on MLE
tf_w, tf_ll, tf_grads, tf_fim, tf_fim_h = session.run([W, log_likelihood, dW, FIM, FIM_h], feed_dict={Xi: Xc, Yi: Yc})

# # RES:

print("TF params: {}".format(tf_w))
print("TF Log-Likelihood: {}".format(tf_ll))
print("TF LL-grad: {}".format(tf_grads))
print("TF FIM:\n {}".format(tf_fim))
print("TF FIM_h:\n {}".format(tf_fim_h))
