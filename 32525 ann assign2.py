#!/usr/bin/env python
# coding: utf-8

# ### ANN Assign2: ANDNOT function using McCulloch-Pitts 

# In[2]:


import numpy as np

def McCulloch_pitts(input,weights,threshold):
  activation = np.dot(input,weights)
  output = 1 if activation >= threshold else 0
  return output

def ANDNOT(x1,x2):
  input = [x1,x2]
  weights = [1,-1]
  threshold = 1
  return McCulloch_pitts(input,weights,threshold)

print(ANDNOT(0,0))
print(ANDNOT(0,1))
print(ANDNOT(1,0))
print(ANDNOT(1,1))


# In[ ]:




