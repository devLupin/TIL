"""
넘파이는 수치 데이터를 다루는 파이썬 패키지
다차원 행렬 자료구조인 ndarray를 통해 벡터 및 행렬을 사용하는 선행 대수 계산에서 주로 사용
속도면에서 우수함.
"""

"""
  Numpy mainly module
1. np.array()   // Generate ndarray from list, tuple, array
2. np.asarray() // Generate ndarray from exsiting array
3. np.arange()  // simillar range()
4. np.linspace(start, end, num)     // Generate n evenly spaced
5. np.logspace(start, end, num)     // Generate n log scale
"""

import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(type(a))
print(a)

#b = np.array([10, 20, 30], [40, 50, 60])   // Only on list in array()
b = np.array([[10, 20, 30], [40, 50, 60]])
print(b)

print(b.ndim)   # dimension
print(b.shape)  # volumn

a = np.zeros((2, 3))    # All elements is 0
#print(a.shape)
print(a)

a = np.ones((2,3))      # All elements is 1
print(a)

n = 7
a = np.full((2, 2), n)  # All elements is n
print(a)

a = np.eye(3)
print(a)
"""
1 0 0
0 1 0
0 0 1
"""

a = np.random.random((n, n))    # All elements is random value
print(a)

a = np.arange(10)
print(a)

a = np.arange(1, 10)    # 1 <= range < 10
print(a)

a = np.arange(1, 10, 2) # 1, 3, 5, 7, 9
print(a)

"""
 1. Generate 1 ~ 29 value in 1d
 2. Reshape 1 ~ 29 value in 2d(5, 6)
"""
a = np.array(np.arange(30).reshape((5, 6)))
print(a)


"""
Numpy slicing
"""

a = np.array([[1, 2, 3], [4, 5, 6]])
b = a[0:2, 0:2]
print(b)

b = a[0, :]     # 0th column elements
print(b)


"""
Numpy integer indexing
"""
a = np.array([[1, 2], [4, 5], [7, 8]])
b = a[[2, 1], [1, 0]]
print(b)    ## [8 4]


"""
Numpy operating
"""

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

b = x + y # == np.add(x, y)
print(b) # == print(x+y)
b = x - y # == np.subtract(x, y)
print(b)
b = x * y # == np.multiply(x, y)
print(b)
b = x / y # == np.devide(x, y)
print(b)

# Must use np.dot() in vector and matrix multiply | matrix multiply
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

c = np.dot(a, b)
print(c)