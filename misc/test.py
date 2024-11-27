import numpy as np

a=np.ones((3,4,5))
print(a[:, (0,1), ..., (0,1)].shape)
b=np.ones((3, 17, 4, 5))
print(b[:, (0,1), ..., (0,1)].shape)