Create a virtual environment
py -m venv .venv

Activate virtual environment
.venv\Scripts\activate

Opening with title, name or alias, date, and short intro describing dataset and objectives. 

Imports
Install seaborn - pip install seaborn
Install scikit-learn - pip install scikit-learn
Section 1: Import and Inspect the Data
# Load Titanic dataset from seaborn and verify
titanic = sns.load_dataset("titanic")
titanic.head()

Section 2: Data Exploration and Preparation
Impute missing values for age using median
Drop rows with missing fare (or impute if preferred)
Create numeric variables (e.g., family_size from sibsp + parch + 1)
Optional - convert categorical features (e.g. sex, embarked) if you think they might help your prediction model. (We do not know relationships until we evaluate things.)

Section 3: Feature Selection and Justification
Define multiple combinations of features to use as inputs to predict fare.
Use unique names (X1, y1, X2, y2, etc.) so results are visible and can be compared at the same time. 
Remember the inputs, usually X, are a 2D array. The target is a 1D array. 

Case 1. age only

# Case 1. age
X1 = titanic[['age']]
y1 = titanic['fare']
 

Case 2. family_size only

# Case 2. family_size
X2 = titanic[['family_size']]
y2 = titanic['fare']

Case 3. age and family size

# Case 3. age, family_size
X3 = titanic[['age', 'family_size']]
y3 = titanic['fare']

# Case 4. pclass
X4 = titanic[['pclass']]
y4 = titanic['fare'

Section 4: Train a Regression Model (Linear Regression)
4.1 Split the Data
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=123)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=123)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=123)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=123)
 
4.2 Train and Evaluate Linear Regression Models (all 4 cases)
lr_model1 = LinearRegression().fit(X1_train, y1_train)
lr_model2 = LinearRegression().fit(X2_train, y2_train)
lr_model3 = LinearRegression().fit(X3_train, y3_train)
lr_model4 = LinearRegression().fit(X4_train, y4_train)

# Predictions
y_pred_train1 = lr_model1.predict(X1_train)
y_pred_test1 = lr_model1.predict(X1_test)
y_pred_train2 = lr_model2.predict(X2_train)
y_pred_test2 = lr_model2.predict(X2_test)

4.3 Report Performance
print("Case 1: Training R²:", r2_score(y1_train, y1_pred_train))
print("Case 1: Test R²:", r2_score(y1_test, y1_pred_test))
print("Case 1: Test RMSE:", mean_squared_error(y1_test, y1_pred_test, squared=False))
print("Case 1: Test MAE:", mean_absolute_error(y1_test, y1_pred_test))

Section 5: Compare Alternative Models (Ridge, Elastic Net, Polynomial Regression)
Regularization adds a penalty to the model’s loss function, discouraging it from using very large weights (coefficients). This makes the model simpler and more likely to generalize well to new data.

In general: 
If the basic linear regression is overfitting, try Ridge.
If you want the model to automatically select the most important features, try Lasso.
If you want a balanced approach, try Elastic Net.

Section 6: Final Thoughts & Insights