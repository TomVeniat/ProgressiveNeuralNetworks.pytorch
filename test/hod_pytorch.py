import numpy as np
# import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.datasets import make_blobs


def vectorize(tensors):
    res = None
    for t in tensors:
        if res is None:
            res = t.view(-1)
        else:
            res = torch.cat([res, t.view(-1)])
    return res


### Create Dataset

np.random.seed(666)
nb_samples = 2000
X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=2, cluster_std=1.1, random_state=2000)

# Transform the original dataset so to learn the bias as any other parameter
Xc = np.ones((nb_samples, X.shape[1] + 1), dtype=np.float32)
Yc = np.zeros((nb_samples, 1), dtype=np.float32)

Xc[:, 0:2] = X
Yc[:, 0] = Y

# ## TF

# In[113]:


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

# In[114]:


# Compute Fisher Information Matrix on MLE
tf_w, tf_ll, tf_grads, tf_fim, tf_fim_h = session.run([W, log_likelihood, dW, FIM, FIM_h], feed_dict={Xi: Xc, Yi: Yc})

# ## Pytorch

# In[5]:


X_py = torch.from_numpy(X).float()
Y_py = Variable(torch.from_numpy(Y)).float().unsqueeze(1)

# In[6]:


n_step = 3500
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0025)

for _ in range(n_step):
    preds = model(Variable(X_py))
    loss = F.binary_cross_entropy(preds.sigmoid(), Y_py, size_average=False) + 0.5
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# In[28]:


pytorch_p = model(Variable(X_py)).sigmoid()
pytorch_w = vectorize(model.parameters())
pytorch_ll = -(Y_py * pytorch_p.log() + (1 - Y_py) * torch.log((1 - pytorch_p) + 1e-9)).sum()
# pytorch_ll = F.binary_cross_entropy(pytorch_p, Y_py, size_average=False)
pytorch_grads = vectorize(torch.autograd.grad(-pytorch_ll, model.parameters(), create_graph=True))
pytorch_fim = torch.mm(pytorch_grads.view(-1, 1), pytorch_grads.view(1, -1))

pytorch_hess_fim = np.array(
    [vectorize(torch.autograd.grad(v, model.parameters(), create_graph=True)).data.numpy() for v in pytorch_grads])

# ## Create a new pytorch model from TF weights

# In[8]:


new_model = nn.Linear(2, 1)
new_model.weight = nn.Parameter(torch.from_numpy(tf_w[:2]).view(1, 2))
new_model.bias = nn.Parameter(torch.from_numpy(tf_w[2]))

# In[15]:


new_p = new_model(Variable(X_py)).sigmoid()

new_pytorch_w = vectorize(new_model.parameters())
new_pytorch_ll = F.binary_cross_entropy(new_p, Y_py, size_average=False)
new_pytorch_grads = vectorize(torch.autograd.grad(-new_pytorch_ll, new_model.parameters(), create_graph=True))
new_pytorch_fim = torch.mm(new_pytorch_grads.view(-1, 1), new_pytorch_grads.view(1, -1))

# new_pytorch_hess_fim = [torch.autograd.grad(v, new_model.parameters()) for v in vectorize(new_pytorch_grads)]


# # RES:

# ### Pytorch

# In[30]:


print("Pytorch params: {}".format(pytorch_w.data.numpy()))
print("Pytorch Log-Likelihood: {}".format(pytorch_ll.data[0]))
print("Pytorch LL-grad: {}".format(pytorch_grads.data.numpy()))
print("Pytorch FIM:\n {}".format(pytorch_fim.data.numpy()))
print("Pytorch FIM (Hess):\n {}".format(pytorch_hess_fim))

# ### TF

# In[115]:


print("TF params: {}".format(tf_w))
print("TF Log-Likelihood: {}".format(tf_ll))
print("TF LL-grad: {}".format(tf_grads))
print("TF FIM:\n {}".format(tf_fim))
print("TF FIM_h:\n {}".format(tf_fim_h))

# ### TF copy (Pytorch B)

# In[12]:


print("New model params: {}".format(new_pytorch_w.data.numpy()))
print("New model Log-Likelihood: {}".format(new_pytorch_ll.data[0]))
print("New model LL-grad: {}".format(new_pytorch_grads.data.numpy()))
print("New model FIM:\n {}".format(new_pytorch_fim.data.numpy()))
