# HOUSE PRICE PREDICTION USING SCIKIT-LEARN
# Dataset: California Housing
# Algorithm: multiple Linear Regression

# Step 1: Import libraries

import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 2: Load California Housing dataset

housing = fetch_california_housing()

print("Dataset loaded successfully")


# Step 3: Convert dataset into DataFrame

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

# Add target column
df["HousePrice"] = housing.target


# Step 4: Display first 5 records

print("\nFirst 5 records:")
print(df.head())


# Step 5: Select only some features

features = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]

X = df[features]

# Target variable
y = df["HousePrice"]


print("\nSelected Input Features:")
print(X.head())

print("\nTarget:")
print(y.head())


# Step 6: Split data into training and testing data
# 75% training and 25% testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=10
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Step 7: Create Linear Regression model

model = LinearRegression()


# Step 8: Train the model

model.fit(X_train, y_train)

print("\nModel training completed")


# Step 9: Predict house prices

predictions = model.predict(X_test)


# Step 10: Display actual and predicted prices

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print("\nActual vs Predicted Prices:")
print(comparison.head(10))


# Step 11: Calculate model performance

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

r2 = r2_score(y_test, predictions)


print("\nModel Evaluation")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# Step 12: Display model coefficients

print("\nModel Coefficients:")

for feature, coefficient in zip(features, model.coef_):
    print(feature, ":", coefficient)

print("\nIntercept:", model.intercept_)


# Step 13: Predict price for a new house

new_house = pd.DataFrame(
    [[
        6.5,     # Median Income
        15.0,    # House Age
        5.5,     # Average Rooms
        2.8      # Average Occupancy
    ]],
    columns=features
)


new_prediction = model.predict(new_house)


print("\nPredicted House Price:")
print(new_prediction[0])

print("\nApproximate Price in US Dollars:")
print("$", new_prediction[0] * 100000)