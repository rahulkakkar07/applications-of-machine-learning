# ============================================================
# A-TO-Z LINEAR REGRESSION USING SCIKIT-LEARN
# Example: Predict Salary based on Years of Experience
# ============================================================


# ------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

# pandas is used for creating and handling tabular data
import pandas as pd

# numpy is used for numerical calculations
import numpy as np

# matplotlib is used for visualization/graphs
import matplotlib.pyplot as plt

# train_test_split is used to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# LinearRegression is the machine learning algorithm we will use
from sklearn.linear_model import LinearRegression

# These are evaluation metrics for regression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ------------------------------------------------------------
# STEP 2: CREATE THE DATASET
# ------------------------------------------------------------

# We are creating a small dataset manually.
#
# Experience = Years of work experience
# Salary     = Salary in thousands

data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [30, 35, 40, 45, 52, 58, 65, 72, 80, 88]
}


# Convert the dictionary into a Pandas DataFrame

df = pd.DataFrame(data)


# ------------------------------------------------------------
# STEP 3: DISPLAY THE DATASET
# ------------------------------------------------------------

print("Original Dataset:")
print(df)


# ------------------------------------------------------------
# STEP 4: UNDERSTAND THE DATASET
# ------------------------------------------------------------

# Display first 5 rows

print("\nFirst 5 rows:")
print(df.head())


# Display last 5 rows

print("\nLast 5 rows:")
print(df.tail())


# Display number of rows and columns

print("\nDataset Shape:")
print(df.shape)


# Display column names

print("\nColumn Names:")
print(df.columns)


# Display information about the dataset

print("\nDataset Information:")
print(df.info())


# Display basic statistical information

print("\nStatistical Summary:")
print(df.describe())


# ------------------------------------------------------------
# STEP 5: CHECK FOR MISSING VALUES
# ------------------------------------------------------------

# isnull() checks whether any value is missing.
# sum() counts the missing values in each column.

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 6: DEFINE INPUT (X) AND OUTPUT (y)
# ------------------------------------------------------------

# X = Input feature(s)
# y = Target/output variable
#
# Here:
# X = Experience
# y = Salary

X = df[['Experience']]

y = df['Salary']


# Display X and y

print("\nInput Features (X):")
print(X)

print("\nTarget Variable (y):")
print(y)


# ------------------------------------------------------------
# STEP 7: CHECK THE SHAPE OF X AND y
# ------------------------------------------------------------

print("\nShape of X:")
print(X.shape)

print("\nShape of y:")
print(y.shape)


# X is 2-dimensional:
# (number_of_rows, number_of_features)
#
# y is generally 1-dimensional:
# (number_of_rows,)


# ------------------------------------------------------------
# STEP 8: VISUALIZE THE ORIGINAL DATA
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

# Scatter plot
plt.scatter(X, y)

# Give title and labels
plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (Thousands)")

# Display the graph
plt.show()


# ------------------------------------------------------------
# STEP 9: SPLIT DATA INTO TRAINING AND TESTING DATA
# ------------------------------------------------------------

# We use 80% data for training
# and 20% data for testing.
#
# random_state=42 ensures that we get the same split
# every time we run the program.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Display training and testing data

print("\nX Training Data:")
print(X_train)

print("\nX Testing Data:")
print(X_test)

print("\ny Training Data:")
print(y_train)

print("\ny Testing Data:")
print(y_test)


# ------------------------------------------------------------
# STEP 10: CHECK THE SIZE OF TRAINING AND TESTING DATA
# ------------------------------------------------------------

print("\nTraining data size:")
print(X_train.shape)

print("\nTesting data size:")
print(X_test.shape)


# ------------------------------------------------------------
# STEP 11: CREATE THE LINEAR REGRESSION MODEL
# ------------------------------------------------------------

# Create an object of LinearRegression

model = LinearRegression()


# At this point:
#
# model = Linear Regression algorithm
#
# But the model has NOT learned anything yet.


# ------------------------------------------------------------
# STEP 12: TRAIN THE MODEL
# ------------------------------------------------------------

# fit() is used to train the machine learning model.
#
# The model learns the relationship between:
#
# Experience ---> Salary

model.fit(X_train, y_train)


# ------------------------------------------------------------
# STEP 13: CHECK THE LEARNED COEFFICIENT
# ------------------------------------------------------------

# coef_ gives us the slope/coefficient of the regression line.

print("\nCoefficient:")
print(model.coef_)


# ------------------------------------------------------------
# STEP 14: CHECK THE INTERCEPT
# ------------------------------------------------------------

# intercept_ gives us the intercept of the regression equation.

print("\nIntercept:")
print(model.intercept_)


# The model has learned an equation:
#
# Salary = Intercept + Coefficient × Experience
#
# For example:
#
# Salary = 21 + 6.5 × Experience
#
# The actual values will depend on our training data.


# ------------------------------------------------------------
# STEP 15: MAKE PREDTIONS ON TEST DATA
# ------------------------------------------------------------

# predict() uses the trained model to make predictions.

y_pred = model.predict(X_test)


# Display actual and predicted values

print("\nActual Salary:")
print(y_test.values)

print("\nPredicted Salary:")
print(y_pred)


# ------------------------------------------------------------
# STEP 16: COMPARE ACTUAL VS PREDICTED VALUES
# ------------------------------------------------------------

# Create a DataFrame for easy comparison

comparison = pd.DataFrame({
    'Experience': X_test['Experience'].values,
    'Actual Salary': y_test.values,
    'Predicted Salary': y_pred
})


print("\nActual vs Predicted Salary:")
print(comparison)


# ------------------------------------------------------------
# STEP 17: CALCULATE MAE
# ------------------------------------------------------------

# MAE = Mean Absolute Error
#
# It tells us the average absolute difference
# between actual and predicted values.

mae = mean_absolute_error(y_test, y_pred)

print("\nMean Absolute Error (MAE):")
print(mae)


# Example:
#
# MAE = 2.5
#
# This means the prediction is off by approximately
# 2.5 salary units on average.


# ------------------------------------------------------------
# STEP 18: CALCULATE MSE
# ------------------------------------------------------------

# MSE = Mean Squared Error
#
# It squares the errors before taking their average.

mse = mean_squared_error(y_test, y_pred)

print("\nMean Squared Error (MSE):")
print(mse)


# ------------------------------------------------------------
# STEP 19: CALCULATE RMSE
# ------------------------------------------------------------

# RMSE = Root Mean Squared Error
#
# RMSE is the square root of MSE.

rmse = np.sqrt(mse)

print("\nRoot Mean Squared Error (RMSE):")
print(rmse)


# ------------------------------------------------------------
# STEP 20: CALCULATE R-SQUARED
# ------------------------------------------------------------

# R² tells us how well the model explains
# the variation in the target variable.

r2 = r2_score(y_test, y_pred)

print("\nR² Score:")
print(r2)


# ------------------------------------------------------------
# STEP 21: DISPLAY ALL EVALUATION METRICS
# ------------------------------------------------------------

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R²   :", r2)


# ------------------------------------------------------------
# STEP 22: PREDICT SALARY FOR A NEW PERSON
# ------------------------------------------------------------

# Suppose a new employee has 11 years of experience.

new_employee = pd.DataFrame({
    'Experience': [11]
})


# Use our trained model to predict salary

new_prediction = model.predict(new_employee)


print("\nPredicted salary for 11 years of experience:")

print(new_prediction)


# ------------------------------------------------------------
# STEP 23: PREDICT FOR MULTIPLE NEW PEOPLE
# ------------------------------------------------------------

# Suppose we have employees with:
#
# 2 years
# 5 years
# 12 years
# 15 years

new_employees = pd.DataFrame({
    'Experience': [2, 5, 12, 15]
})


# Make predictions

new_predictions = model.predict(new_employees)


print("\nPredictions for new employees:")

for experience, salary in zip(
    new_employees['Experience'],
    new_predictions
):
    print(
        f"{experience} years experience "
        f"-> predicted salary = {salary:.2f}"
    )


# ------------------------------------------------------------
# STEP 24: VISUALIZE THE REGRESSION LINE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))


# Plot original data points

plt.scatter(
    X,
    y,
    label="Actual Data"
)


# Plot regression line
#
# model.predict(X) calculates predicted salary
# for every experience value in X.

plt.plot(
    X,
    model.predict(X),
    label="Regression Line"
)


# Add title

plt.title("Linear Regression: Experience vs Salary")


# Add axis labels

plt.xlabel("Years of Experience")
plt.ylabel("Salary (Thousands)")


# Display legend

plt.legend()


# Display graph

plt.show()


# ------------------------------------------------------------
# STEP 25: PRINT THE FINAL REGRESSION EQUATION
# ------------------------------------------------------------

coefficient = model.coef_[0]

intercept = model.intercept_


print("\nRegression Equation:")

print(
    f"Salary = {intercept:.2f} + "
    f"{coefficient:.2f} × Experience"
)


# ============================================================
# END OF PROGRAM
# ============================================================