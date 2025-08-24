🏡 Predicting California Housing Prices
📌 Project Overview

This project builds a machine learning model to predict median house prices in California districts based on census data. 

The goal is to explore the full ML workflow, from data exploration and preprocessing to model training, evaluation.

📂 Dataset

The dataset contains information about different districts in California, including:

longitude, latitude → location

housing_median_age → median age of houses

total_rooms, total_bedrooms → housing stock

population, households → demographics

median_income → income level of residents

median_house_value → target variable (house price to predict)

🛠️ Workflow

Exploratory Data Analysis (EDA)

Visualized distributions, correlations, and geographic housing price trends.

Identified missing values and outliers.

Data Preprocessing

Handled missing values with imputation.

Feature scaling with StandardScaler.

Encoded categorical features (ocean_proximity) using OneHotEncoder.

Added engineered features (e.g., rooms_per_household, bedrooms_per_room).

Model Training

Baseline model: Linear Regression.

Nonlinear models: Decision Tree, Random Forest.

Evaluated using cross-validation.

Model Evaluation

Metrics used: RMSE, MAE, MAPE.

Compared baseline vs tuned models.

Visualized predicted vs actual prices and error distributions.
 

📊 Results

Linear Regression provided a strong baseline.

Decision Tree overfit the training data (high variance).

Random Forest (tuned) achieved the best performance, reducing error significantly compared to baseline.

Final model achieved ~17.75% MAPE, meaning predictions were usually within 20% of actual house values.

🚀 Tech-Stack Used

Python 🐍

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

📌 Key Learnings

Importance of feature engineering for boosting model performance.

Why cross-validation is essential to avoid overfitting.

Trade-offs between simple interpretable models (Linear Regression) and complex ensemble models (Random Forest).
