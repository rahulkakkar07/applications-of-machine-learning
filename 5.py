# ============================================================
# IRIS FLOWER CLASSIFICATION USING LOGISTIC REGRESSION
# ============================================================


# ------------------------------------------------------------
# STEP 1: Import Required Libraries
# ------------------------------------------------------------

# Pandas is used for handling data in table form
import pandas as pd

# Matplotlib and Seaborn are used for graphs
import matplotlib.pyplot as plt
import seaborn as sns

# Load the built-in Iris dataset
from sklearn.datasets import load_iris

# Used to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# Used for feature scaling
from sklearn.preprocessing import StandardScaler

# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# STEP 2: Load the Iris Dataset
# ============================================================

iris = load_iris()


# ============================================================
# STEP 3: Convert Dataset into Pandas DataFrame
# ============================================================

# iris.data contains the four input features
# iris.feature_names contains names of those features

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target column
# 0 = Setosa
# 1 = Versicolor
# 2 = Virginica

df["target"] = iris.target


# Add species names for better understanding
df["species"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})


# ============================================================
# STEP 4: Display First 5 Records
# ============================================================

print("First 5 Records:")
print(df.head())


# ============================================================
# STEP 5: Separate Input Features and Output
# ============================================================

# X contains input features:
# Sepal Length
# Sepal Width
# Petal Length
# Petal Width

X = df[iris.feature_names]

# y contains the target/output
# 0, 1, or 2

y = df["target"]


print("\nInput Features:")
print(X.head())

print("\nTarget Values:")
print(y.head())


# ============================================================
# STEP 6: Split Dataset into Training and Testing Sets
# ============================================================

# 80% data will be used for training
# 20% data will be used for testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,      # 20% testing data
    random_state=42,     # same result every time
    stratify=y           # maintains class distribution
)


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# STEP 7: Feature Scaling
# ============================================================

# StandardScaler transforms the features so that they
# have approximately:
#
# Mean = 0
# Standard Deviation = 1

scaler = StandardScaler()


# Fit the scaler only on training data
# and transform the training data

X_train_scaled = scaler.fit_transform(X_train)


# Use the same scaling parameters for testing data

X_test_scaled = scaler.transform(X_test)


print("\nFirst 5 Scaled Training Records:")
print(X_train_scaled[:5])


# ============================================================
# STEP 8: Create Logistic Regression Model
# ============================================================

model = LogisticRegression(
    max_iter=200
)


# ============================================================
# STEP 9: Train the Model
# ============================================================

# fit() allows the model to learn patterns
# from the training data

model.fit(
    X_train_scaled,
    y_train
)


print("\nModel Training Completed Successfully!")


# ============================================================
# STEP 10: Predict Classes for Testing Data
# ============================================================

# predict() gives the predicted class
# for every test sample

y_pred = model.predict(
    X_test_scaled
)


print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)


# ============================================================
# STEP 11: Calculate Accuracy
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nModel Accuracy:")
print(accuracy)


print("\nModel Accuracy in Percentage:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# STEP 12: Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion Matrix:")
print(cm)


# ============================================================
# STEP 13: Display Confusion Matrix as Heatmap
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,                  # show values inside boxes
    fmt="d",                     # display integers
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")

plt.show()


# ============================================================
# STEP 14: Classification Report
# ============================================================

# Classification report gives:
#
# Precision
# Recall
# F1-score
# Support

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ============================================================
# STEP 15: Predict Species of a New Flower
# ============================================================

# Suppose a new flower has:
#
# Sepal Length = 5.1 cm
# Sepal Width  = 3.5 cm
# Petal Length = 1.4 cm
# Petal Width  = 0.2 cm


new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=iris.feature_names
)


print("\nNew Flower Data:")
print(new_flower)


# ============================================================
# STEP 16: Scale the New Flower Data
# ============================================================

# We must use the SAME scaler that was used
# for the training data

new_flower_scaled = scaler.transform(
    new_flower
)


# ============================================================
# STEP 17: Predict New Flower Species
# ============================================================

prediction = model.predict(
    new_flower_scaled
)


print("\nPredicted Class Number:")
print(prediction[0])


# Convert class number to flower species name

predicted_species = iris.target_names[
    prediction[0]
]


print("\nPredicted Flower Species:")
print(predicted_species)


# ============================================================
# STEP 18: Predict Probability of Each Species
# ============================================================

probabilities = model.predict_proba(
    new_flower_scaled
)


print("\nPrediction Probabilities:")

for species, probability in zip(
    iris.target_names,
    probabilities[0]
):
    print(f"{species}: {probability:.4f}")


# ============================================================
# END OF PROGRAM
# ============================================================