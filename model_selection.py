import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the dataset
housing=pd.read_csv('housing.csv')

# 2. create stratified train and test set
housing['income_category']=pd.cut(housing['median_income'], bins=[0.0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1,2,3,4,5])


split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_indices, test_indices in split.split(housing, housing['income_category']):
    train_set=housing.loc[train_indices].drop('income_category', axis=1) # we will only work on training set
    test_set=housing.loc[test_indices].drop('income_category', axis=1) # put aside the test set

# We will work on the copy of training data
housing=train_set.copy()

# 3. Separate Features and labels
housing_labels=housing['median_house_value'].copy() # --> label
housing=housing.drop('median_house_value', axis=1)  # --> Features 
# print(housing, housing_labels)

# 4. Separate numerical and categorical columns
num_atrributes= housing.drop('ocean_proximity', axis=1).columns.tolist()
cat_attributes=['ocean_proximity']

# 5. Now let's make the pipeline for numerical columns
numerical_pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 6. Making pipeline for categorical column
cat_pipeline=Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))   # we also have handle_unknown = 'error' which throws error when a value is given to which was not in training set     
])

# 7. Construct a full Pipeline

full_pipeline=ColumnTransformer([
    ('num', numerical_pipeline, num_atrributes),
    ('cat', cat_pipeline, cat_attributes)
])

# 8. Transfrom the data
housing_prepared=full_pipeline.fit_transform(housing)
 
# 9. Train the model

# Linear Regression Model
lin_reg_model=LinearRegression()
lin_reg_model.fit(housing_prepared, housing_labels)
lin_predict=lin_reg_model.predict(housing_prepared)
lin_rmse=root_mean_squared_error(housing_labels,lin_predict)
print(f"Linear Regression root mean square error(without cross validation): {lin_rmse}")

lin_rmses=-cross_val_score(LinearRegression(), housing_prepared, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
print(f"Linear Regression root mean square error(with cross validation): {pd.Series(lin_rmses).describe()}")
# Decision Tree
dec_tree_model=DecisionTreeRegressor()
dec_tree_model.fit(housing_prepared, housing_labels)
dec_tree_predict=dec_tree_model.predict(housing_prepared)
dec_tree_rmse=root_mean_squared_error(housing_labels, dec_tree_predict)
print(f"Decision Tree root mean square error( without Cross validation ): {dec_tree_rmse}")  #---> In this case it will just overfit the data

# Cross Validation
dec_rmses=-cross_val_score(DecisionTreeRegressor(), housing_prepared, housing_labels, scoring='neg_root_mean_squared_error', cv=10)

print(f"Decision Tree root mean square error( with Cross validation ): ",pd.Series(dec_rmses).describe())

# Random Forest Regressor
ran_forest_model=RandomForestRegressor()
ran_forest_model.fit(housing_prepared, housing_labels)
ran_predict=ran_forest_model.predict(housing_prepared)
ran_rmse=root_mean_squared_error(housing_labels, ran_predict)
print(f"Random Forest Regressor root mean square error( without Cross validation ): {ran_rmse}")

ran_rmses=cross_val_score(RandomForestRegressor(), housing_prepared, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
print(f"Random Forest Regressor root mean square error( with Cross validation ): {pd.Series(ran_rmses).describe()}")