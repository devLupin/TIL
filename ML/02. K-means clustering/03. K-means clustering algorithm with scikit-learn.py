from sklearn.datasets import make_blobs
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

#가상의 데이터 셋 생성
x, y = make_blobs(n_samples=100, centers=4, n_features=2, random_state=6)
points = pd.DataFrame(x, y).reset_index(drop=True)
points.columns = ["x", "y"]
points.head()

kmeans = KMeans(n_clusters=4)
kmeans.fit(points)

result = points.copy()
result["cluster"] = kmeans.labels_
print(result.head())

sns.scatterplot(x="x", y="y", hue="cluster", data=result, palette="Set2");