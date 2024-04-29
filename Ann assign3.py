#!/usr/bin/env python
# coding: utf-8

# ### ANN assign3:recognise even and odd numbers
# 

# In[1]:


import numpy as np

#training data
training_inputs = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

weights = np.zeros(10)
bias = np.random.rand()
learning_rate = 0.1

for _ in range(100):
    for inputs, label in zip(training_inputs, labels):
        prediction = 1 if np.dot(inputs, weights)+bias >= 0 else 0
        weights += learning_rate * (label - prediction) * inputs

# Testing data
test_inputs = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
])

for inputs in test_inputs:
    prediction = 1 if np.dot(inputs, weights) >= 0 else 0
    number = np.argmax(inputs)
    print(f"ASCII {number}: {'Odd' if prediction == 1 else 'Even'}")


# In[ ]:




