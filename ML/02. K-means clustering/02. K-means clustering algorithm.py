#%%
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import math
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("Set2")
# %%
#가상의 데이터 셋 생성
x, y = make_blobs(n_samples=100, centers=4, n_features=2, random_state=6)
points = pd.DataFrame(x, y).reset_index(drop=True)
points.columns = ["x", "y"]
points.head()
# %%
#중심점 개수 탐색을 위한 드로윙
sns.scatterplot(x="x", y="y", data=points, palette="Set2")
# %%
# 중심점(centroid)
# x, y 좌표의 평균
# 시작 단계라서, cluster의 중심점을 구할 수 없어 전체 데이터 중, 랜덤한 k개의 데이터를 중심점으로 사용
centorids = points.sample(4, random_state=1)
centorids
# %%
# 각 데이터에 대해 4 개의 중심점과의 거리 계산
# 그 후, 가장 가까운 중심점의 cluster로 데이터 할당

# 각 데이터 유클리드 거리 계산
distance = sp.spatial.distance.cdist(points, centorids, "euclidean")

# 가장 거리가 짧은 중심점의 cluster로 할당
cluster_num = np.argmin(distance, axis=1)

result = points.copy()
result["cluster"] = np.array(cluster_num)
result.head()
# %%
# cluster 별로 다른 색 부여
sns.scatterplot(x="x", y="y", hue="cluster", data=result, palette="Set2")
# %%
# 클러스터의 중심점 : x, y 평균
centorids_2 = result.groupby("cluster").mean()
centorids_2
# %%
distance_2 = sp.spatial.distance.cdist(points, centorids_2, "euclidean")

cluster_num_2 = np.argmin(distance, axis=1)

result_2 = points.copy()
result_2["cluster"] = np.array(cluster_num_2)
result_2.head()
sns.scatterplot(x="x", y="y", hue="cluster", data=result_2, palette="Set2")