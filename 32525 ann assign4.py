#!/usr/bin/env python
# coding: utf-8

# ### ANN assign4 : perceptron learning law 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Define training data
X_train = np.array([[2, 4], [4, 3], [5, 6], [7, 5], [9, 4], [6, 2], [8, 3]])
y_train = np.array([1, 1, 1, 1, -1, -1, -1])

# Initialize weights and bias randomly  
weights = np.random.rand(2)
bias = np.random.rand()

# Training the perceptron
learning_rate = 0.1
num_epochs = 100

for _ in range(num_epochs):
    for inputs, label in zip(X_train, y_train):
        summation = np.dot(weights, inputs) + bias
        activation = 1 if summation >= 0 else -1
        weights += learning_rate * (label - activation) * inputs
        bias += learning_rate * (label - activation)

# Decision regions plot
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')

x = np.linspace(0, 10, 100)
y = -(weights[0] * x + bias) / weights[1]
plt.plot(x, y, color='red', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Regions')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




