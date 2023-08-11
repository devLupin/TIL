#%%
import pandas as pd

file_path = './melb_data.csv'
data = pd.read_csv(file_path)

filtered_data = data.dropna(axis=0);

y = filtered_data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_data[features]

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X, y)

from sklearn.metrics import mean_absolute_error
predict_price = model.predict(X)
MAE = mean_absolute_error(y, predict_price)
print(MAE)


#%%
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

model = DecisionTreeRegressor()
model.fit(train_X, train_y)

predict = model.predict(val_X)
print(mean_absolute_error(val_y, predict))