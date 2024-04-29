#!/usr/bin/env python
# coding: utf-8

# ### ANN assign1: Activation Functions

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def binaryStep(x):
    ''' It returns '0' is the input is less then zero otherwise it returns one '''
    return np.heaviside(x,1)

def linear(x):
    ''' y = f (x) It returns the input as it is'''
    return x

def sigmoid(x):
    ''' It returns 1/ (1+exp (-x)). where the values lies between zero and one '''
    return 1/(1+np.exp(-x))

def tanh(x):
    ''' It returns the value (1-exp (-2x))/ (1+exp (-2x)) and the value returned will be lies in between -1 to 1.'''
    return np.tanh(x)

def RELU(x):
    ''' It returns zero if the input is less than zero otherwise it returns the given input. '''
    x1=[]
    for i in x:
        if i<0:
            x1.append(0)
        else:
            x1.append(i)
    return x1

x = np.linspace(-10, 10)

plt.plot(x, binaryStep(x), label='Binary Step')
plt.plot(x, linear(x), label='Linear')
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.plot(x, RELU(x), label='RELU')

plt.title('Activation Functions')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


# In[ ]:




