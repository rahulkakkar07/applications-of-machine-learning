# ==========================================
# Iris Dataset - Basic Data Checking
# ==========================================

# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert the feature data into a Pandas DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target column
df["target"] = iris.target

# Add species names for easy understanding
df["species"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

# Display first 5 rows
print("First 5 Records:")
print(df.head())

# Check dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display feature names
print("\nFeature Names:")
print(iris.feature_names)

# Display class names
print("\nClass Names:")
print(iris.target_names)

# Check data types
print("\nData Types:")
print(df.dtypes)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check class distribution
print("\nClass Distribution:")
print(df["species"].value_counts())