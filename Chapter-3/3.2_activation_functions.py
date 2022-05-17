import torch
import matplotlib.pyplot as plt

x = torch.range(-5., 5., 0.1)

# Sigmoid
y = torch.sigmoid(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

# tanh
y = torch.tanh(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

# ReLU
y = torch.relu(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

# PReLU
prelu = torch.nn.PReLU(num_parameters=1)
y = prelu(x)
plt.plot(torch.detach(x).numpy(), torch.detach(y).numpy())
plt.show()

# softmax
import torch.nn as nn

softmax = nn.Softmax(dim=1)
x_input = torch.randn(1, 3)
y_input = softmax(x_input)
print(x_input)
print(y_input)
print(torch.sum(y_input, dim=1))

