import numpy as np
import matplotlib.pyplot as mlib
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

data_set = pd.read_csv("Housing.csv")
feature_names = ['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea', 'furnishingstatus']

x = data_set.drop(['price'], axis=1)
y = data_set['price']

le = LabelEncoder()
x['mainroad'] = le.fit_transform(x['mainroad'])
x['guestroom'] = le.fit_transform(x['guestroom'])
x['basement'] = le.fit_transform(x['basement'])
x['hotwaterheating'] = le.fit_transform(x['hotwaterheating'])
x['prefarea'] = le.fit_transform(x['prefarea'])
x['airconditioning'] = le.fit_transform(x['airconditioning'])
x['furnishingstatus'] = le.fit_transform(x['furnishingstatus'])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)
train_data = train_x.join(train_y)

# creating histograms
hgram = pd.DataFrame(x)
hgram.columns = feature_names
fig = mlib.figure(figsize=(10,8))
ax = fig.gca()
hgram.hist(ax=ax)

# creating a heat map to visualize correlation between features better
mlib.figure(figsize=(10,8))
sb.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')

train_data['area'] = np.log(train_data['area'] + 1)
train_data['bedrooms'] = np.log(train_data['bedrooms'] + 1)
train_data['bathrooms'] = np.log(train_data['bathrooms'] + 1)
train_data['stories'] = np.log(train_data['stories'] + 1)
train_data['parking'] = np.log(train_data['parking'] + 1)

mlib.show()
scaler = StandardScaler()

train_x, train_y = train_data.drop(['price'], axis=1), train_data['price']
train_x_scaled = scaler.fit_transform(train_x)

# Using Linear Regression
lReg = LinearRegression()
lReg.fit(train_x, train_y)

test_data = test_x.join(test_y)
test_data['area'] = np.log(test_data['area'] + 1)
test_data['bedrooms'] = np.log(test_data['bedrooms'] + 1)
test_data['bathrooms'] = np.log(test_data['bathrooms'] + 1)
test_data['stories'] = np.log(test_data['stories'] + 1)
test_data['parking'] = np.log(test_data['parking'] + 1)

test_x, test_y = test_data.drop(['price'], axis=1), test_data['price']
test_x_scaled = scaler.fit_transform(test_x)
print("Linear Regression Score:", lReg.score(test_x, test_y))

# Using Random Forest Regression
forest = RandomForestRegressor()
forest.fit(train_x, train_y)
print("Random Forest Regression Score:", forest.score(test_x, test_y))

# Using Grid Search for a more comprehensive comparison between the features
forest = RandomForestRegressor()
param_grid = {
    "n_estimators": [30, 80, 100],
    "min_samples_split": [2, 4],
    "max_depth": [None, 4, 8]
}
grid_search = GridSearchCV(forest, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(train_x, train_y)
best_forest = grid_search.best_estimator_
print("GridSearchCV score:", best_forest.score(test_x, test_y))

# Representing the Linear Regression data using a scatter plot
colors = np.random.rand(137)
mlib.scatter(test_y, lReg.predict(test_x), c=colors, alpha=0.7, cmap='viridis')
mlib.title("Test Prices vs Predicted Prices")
mlib.xlabel("Test Prices")
mlib.ylabel("Predicted Prices")
mlib.colorbar(label='Color Intensity')

mlib.show()