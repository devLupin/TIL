"""
  Pandas mainly used
1. Series
2. DataFrame    // most
3. Panel
"""

import pandas as pd

sr = pd.Series([17000, 18000, 1000, 5000],
        index=["피자", "치킨", "콜라", "맥주"])
print(sr)
print("\n")
print(sr.values)
print("\n")
print(sr.index)


values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index=index, columns=columns)
print(df)
print(df.index)
print(df.columns)
print(df.values)

data = [
    ['1000', 'Steve', 90.72],
    ['1001', 'James', 78.09],
    ['1002', 'Doyeon', 98.43],
    ['1003', 'Jane', 64.19],
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
]
df = pd.DataFrame(data, columns=['학번', '이름', '점수'])


data = { '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
'이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
         '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]}
df = pd.DataFrame(data)
print(df)

df.head(n) # front number of n
df.tail(n) # rear number of n
df['column'] 

print(df.head(3))
print(df.tail(3))
print(df['학번'])

df=pd.read_csv('example.csv')  # external file read
print(df)
print(df.index)