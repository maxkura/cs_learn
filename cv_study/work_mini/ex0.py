import cv2
import numpy as np
x=np.array([[1,2],[3,4]])
y=x+4
print('x')
print(x)
print('y')
print(y)
print('x和y主逐元素相乘')
print(np.multiply(x,y))
print('x和y矩阵相乘')
print(x@y)

