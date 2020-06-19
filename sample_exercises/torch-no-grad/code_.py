import numpy as np
import matplotlib.pyplot as plt

import torch

# ensure reproducibility
torch.manual_seed(0)

# initialize X and y
n = 1
m = 100
X = torch.rand(size=(m, n))
y = torch.matmul(X, 10*torch.ones(n, 1)) + 1.5*torch.rand(size=(m, 1))

# define hyperparameters
lr = 0.01
epochs = 10000

# initialize weights, bias
theta = torch.rand(size=(n, 1))
b = torch.rand(size=(1, 1))

# perform gradient descent
losses = list()
for epoch in range(epochs):
    h = torch.matmul(X, theta) + b
    theta_grad = (1 / m) * torch.matmul(X.T, (h - y))
    b_grad = (1 / m) * torch.sum(h - y)

    loss = (1 / (2*m)) * torch.matmul((h - y).T, (h - y))   
    losses.append(loss.item())
    theta -= lr*theta_grad
    b -= lr*b_grad

# plot loss function
plt.plot(losses)
plt.xlabel("Epoch"); plt.ylabel("loss")

# perform predictions on data
y_ = torch.matmul(X, theta) + b

# plot data with predicted values
plt.scatter(X, y, alpha=0.5)
plt.plot(X, y_, color="red")
plt.legend(["predicted", "data"])
plt.xlabel("X"); plt.ylabel("y")
