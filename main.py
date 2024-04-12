import sklearn as sk
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# import csv file
housing_data = pd.read_csv("C:/Users/jessh/OneDrive/HousePricePrediction/Data/American_Housing_Data_20231209.csv")

# drops missing values
housing_data = housing_data.dropna(axis=0)

# prediction target is the price
y = housing_data.Price

# features
features = ['Zip Code', 'Beds', 'Baths', 'Living Space', 'Median Household Income']
X = housing_data[features]

# define train and test values
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# define model
# housing_model = DecisionTreeRegressor()

# fit model
# housing_model.fit(train_X, train_y)

# gets predicted prices on validation data
# val_predictions = housing_model.predict(val_X)

# checks mean absolute error
# mae = mean_absolute_error(val_y, val_predictions)
# print(mae) # printed 189209.879


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    housing_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    housing_model.fit(train_X, train_y)
    val_prediction = housing_model.predict(val_X)
    mae = mean_absolute_error(val_y, val_prediction)
    return mae 

for max_leaf_nodes in [5, 50, 500, 2000, 2500, 5000, 7500, 10000]:
    new_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    # print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, new_mae))
    # seems like 2500 is optimal because its mae is 181463 

# using random forest instead
forest_model = RandomForestRegressor( random_state=1)
forest_model.fit(train_X, train_y)
forest_predictions = forest_model.predict(val_X) 
# print(mean_absolute_error(val_y, forest_predictions)) # better now it's 148950 

def get_mae_forest(max_leaf_nodes, train_X, val_X, train_y, val_y):
    forest_model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    forest_model.fit(train_X, train_y)
    val_prediction = forest_model.predict(val_X)
    mae2 = mean_absolute_error(val_y, val_prediction)
    return mae2 

# testing max leaf nodes
for max_leaf_nodes in [9400, 9500, 9600]:
    new_mae_forest = get_mae_forest(max_leaf_nodes, train_X, val_X, train_y, val_y)
    # print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, new_mae_forest))
    # 9500 seems to be the best value approx. mae = 149635 

# testing number of random trees produced

def get_mae_forest1(n_estimators, max_leaf_nodes, train_X, val_X, train_y, val_y):
    forest_model = RandomForestRegressor(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=1)
    forest_model.fit(train_X, train_y)
    val_prediction = forest_model.predict(val_X)
    mae2 = mean_absolute_error(val_y, val_prediction)
    return mae2 


for n_estimators in [150, 500]:
    new_mae_forest = get_mae_forest1(n_estimators, 9500, train_X, val_X, train_y, val_y)
    # print("Number of trees: %d \t\t Mean Absolute Error: %d" %(n_estimators, new_mae_forest))
    # time is quite bad, should stick with 100 or 150 trees 



final_model = RandomForestRegressor(n_estimators=100, max_leaf_nodes=9500, random_state=1)
final_model.fit(train_X, train_y)
predictions = final_model.predict(val_X)
final_mae = mean_absolute_error(predictions, val_y)
final_mape = mean_absolute_percentage_error(predictions, val_y)
print(final_mae)
print(final_mape)

# NOTES:
# with 150 trees mae is 149053
# with 100 trees mae is 149635
# percent error is 21% or .21529
