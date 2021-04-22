# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:35:54 2021

@author: Maëlys
"""


import numpy as np

a = np.empty((0, 2))
a

a = np.append(a, np.array([[1,2]]),axis=0) 
a
a.dim

import inspect



from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import lasso_path

lines = inspect.getsource(lasso_path)
print(lines)