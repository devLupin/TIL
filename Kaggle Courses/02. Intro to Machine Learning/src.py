#%%
import pandas as pd

#%%
file_path = '../../../melb_data.csv'
data = pd.read_csv(file_path)
data.describe()
# %%
data.columns
# %%
data = data.dropna(axis=0)
data.columns
# %%
data.describe
# %%
data.describe()
# %%
y = data.Price
y
# %%
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# %%
X = data[features]
# %%
X.describe()
# %%
X.head()
# %%
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)
# %%
print(X.head())
print(model.predict(X.head()))
print(y.head())
# %%
