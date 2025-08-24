import os
import joblib
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
from sklearn.metrics import mean_absolute_error

MODEL_FILE='model.pkl'
PIPELINE_FILE='pipeline.pkl'

def build_pipeline(nums_attribs, cat_attribs):
    num_pipeline=Pipeline([
        ('Impute', SimpleImputer(strategy='mean')),
        ('Scaler', StandardScaler())
    ])

    cat_pipeline=Pipeline([
        ('Encoder', OneHotEncoder())
    ])

    final_pipeline=ColumnTransformer([
        ('num', num_pipeline, nums_attribs),
        ('cat', cat_pipeline, cat_attribs)
    ])

    return final_pipeline

if not os.path.exists(MODEL_FILE):
    # lets train the model
    df=pd.read_csv('housing.csv')
    df['income_category']=pd.cut(df['median_income'], bins=[0,1.5,3,4.5,6,np.inf], labels=[1,2,3,4,5])

    strat_split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_indices, test_indices in strat_split.split(df, df['income_category']):
        strat_train_set=df.loc[train_indices]
        strat_test_set=df.loc[test_indices]

    train_set=strat_train_set.copy()
    train_set.drop('income_category', inplace=True, axis=1)

    strat_test_set.drop('income_category', axis=1).to_csv("test_set.csv", index=False)
    test_set=strat_test_set.copy()
    test_set.drop(columns=['income_category', 'median_house_value'], inplace=True, axis=1)
    test_set.to_csv('input.csv', index=False)

    features=train_set.drop('median_house_value',axis=1)
    label=train_set['median_house_value']

    num_attribs=list(features.drop('ocean_proximity', axis=1).columns)
    cat_attribs=['ocean_proximity']

    pipeline=build_pipeline(num_attribs, cat_attribs)
    features=pipeline.fit_transform(features)
    
    model=RandomForestRegressor(random_state=42)
    model.fit(features, label)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is Trained")

else:
    # Lets do inference
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)

    df=pd.read_csv('test_set.csv')

    input_data=pd.read_csv('input.csv')
    transformed_input=pipeline.transform(input_data)
    predictions=model.predict(transformed_input)
    input_data['median_house_value (Predictions)']=predictions
    input_data['median_house_value (Actual)']=df['median_house_value']
    input_data.to_csv('output.csv', index=False)

    print("Inference is completed, results saved to output.csv, Enjoy!!!")
    mape = np.mean(np.abs((df['median_house_value'] - predictions) / df['median_house_value'])) * 100
    print("Mean % Error:", mape, "%")
    Rmse=root_mean_squared_error(df['median_house_value'], predictions)
    print(f"Rmse: {Rmse}")
    mae=mean_absolute_error(df['median_house_value'], predictions)
    print(f"mae: {mae}")
