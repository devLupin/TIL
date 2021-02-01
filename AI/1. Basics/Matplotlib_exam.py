"""
Matplotlib use for data visualaization as chart or plot

Matplotlib usage
1. Before data analysis, visualaization for data understanding
2. After data analysis, result visualaization   
"""

import matplotlib.pyplot as plt

#%%
import matplotlib.pyplot as plt
plt.title('test')
plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
plt.show()
# %%
plt.title('test')
plt.plot([1,2,3,4],[2,4,8,6])
plt.show()
# %%
plt.plot([1, 2, 3, 4])
plt.show()
# %%
plt.title('test')
plt.plot([1,2,3,4],[2,4,8,6])
plt.show()
# %%
plt.title('students')
plt.plot([1,2,3,4],[2,4,8,6])
plt.plot([1.5,2.5,3.5,4.5],[3,5,8,10]) #라인 새로 추가
plt.xlabel('hours')
plt.ylabel('score')
plt.legend(['A student', 'B student']) #범례 삽입
plt.show()
# %%
