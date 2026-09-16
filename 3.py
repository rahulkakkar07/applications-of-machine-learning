# ============================================================
# DATA PREPROCESSING AND EXPLORATORY DATA ANALYSIS
# COMPLETE PRACTICAL SOLUTION
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# Improve graph appearance
sns.set_style("whitegrid")


# ============================================================
# STEP 2: CREATE THE DATASET
# ============================================================

data = {
    "Employee_ID": [
        101, 102, 103, 104, 105,
        106, 107, 108, 109, 110,
        111, 112, 113, 114, 115,
        116, 117, 118, 119, 120,
        121, 122, 123, 124, 125,
        126, 127, 128, 129, 129
    ],

    "Age": [
        25, 30, np.nan, 40, 35,
        29, 31, 120, 45, 38,
        26, 33, 42, 36, 28,
        50, 41, 34, 39, 32,
        27, 44, 37, 48, 30,
        52, 46, 29, 35, 35
    ],

    "Gender": [
        "Male", "Female", "female", "Male", "M",
        "Female", "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "female", "Male", "Male"
    ],

    "Department": [
        "IT", "HR", "IT", "Finance", "HR",
        "Finance", np.nan, "IT", "Finance", "HR",
        "IT", "Finance", "HR", "IT", "Finance",
        "HR", "IT", "Finance", "HR", "IT",
        "Finance", "HR", "IT", "Finance", "HR",
        "IT", "Finance", "HR", "IT", "IT"
    ],

    "Experience": [
        2, 5, 4, 12, 8,
        3, 6, 10, 15, 9,
        2, 7, 13, 8, 4,
        20, 12, 7, 10, 6,
        3, 14, 9, 17, 5,
        22, 16, 4, 8, 8
    ],

    "Salary": [
        30000, 45000, 40000, 70000, np.nan,
        38000, 50000, 500000, 85000, 60000,
        32000, 55000, 75000, 58000, 36000,
        95000, 72000, 52000, 65000, 48000,
        34000, 80000, 62000, 90000, 44000,
        100000, 88000, 39000, 57000, 57000
    ],

    "Purchased_Course": [
        "No", "Yes", "No", "Yes", "Yes",
        "No", "Yes", "Yes", "Yes", "No",
        "No", "Yes", "Yes", "No", "No",
        "Yes", "Yes", "No", "Yes", "No",
        "No", "Yes", "No", "Yes", "No",
        "Yes", "Yes", "No", "Yes", "Yes"
    ]
}

df = pd.DataFrame(data)


# ============================================================
# STEP 3: DISPLAY THE ORIGINAL DATASET
# ============================================================

print("\nORIGINAL DATASET")
print("=" * 70)
print(df)


# ============================================================
# STEP 4: INITIAL DATA INSPECTION
# ============================================================

print("\nFIRST FIVE ROWS")
print("=" * 70)
print(df.head())


print("\nLAST FIVE ROWS")
print("=" * 70)
print(df.tail())


print("\nDATASET SHAPE")
print("=" * 70)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


print("\nCOLUMN NAMES")
print("=" * 70)
print(df.columns.tolist())


print("\nDATA TYPES")
print("=" * 70)
print(df.dtypes)


print("\nDATASET INFORMATION")
print("=" * 70)
df.info()


print("\nSTATISTICAL SUMMARY")
print("=" * 70)
print(df.describe())


print("\nCOMPLETE SUMMARY")
print("=" * 70)
print(df.describe(include="all"))


# ============================================================
# STEP 5: CHECK DUPLICATE RECORDS
# ============================================================

print("\nNUMBER OF COMPLETELY DUPLICATED ROWS")
print("=" * 70)
print(df.duplicated().sum())


print("\nDUPLICATED EMPLOYEE IDs")
print("=" * 70)

duplicate_employee_ids = df[
    df.duplicated(
        subset="Employee_ID",
        keep=False
    )
]

print(duplicate_employee_ids)


# ============================================================
# STEP 6: REMOVE DUPLICATE EMPLOYEE RECORDS
# ============================================================

df = df.drop_duplicates(
    subset="Employee_ID",
    keep="first"
).copy()

print("\nSHAPE AFTER REMOVING DUPLICATE EMPLOYEE IDs")
print("=" * 70)
print(df.shape)


# ============================================================
# STEP 7: CHECK UNIQUE CATEGORICAL VALUES
# ============================================================

print("\nORIGINAL GENDER CATEGORIES")
print("=" * 70)
print(df["Gender"].unique())


print("\nORIGINAL DEPARTMENT CATEGORIES")
print("=" * 70)
print(df["Department"].unique())


# ============================================================
# STEP 8: STANDARDIZE INCONSISTENT CATEGORICAL VALUES
# ============================================================

# Remove spaces and convert gender values to lowercase
df["Gender"] = (
    df["Gender"]
    .str.strip()
    .str.lower()
)

# Replace abbreviated values
df["Gender"] = df["Gender"].replace({
    "m": "male",
    "f": "female"
})

# Remove spaces and convert departments to title case
df["Department"] = (
    df["Department"]
    .str.strip()
    .str.title()
)


print("\nCLEANED GENDER CATEGORIES")
print("=" * 70)
print(df["Gender"].unique())


print("\nCLEANED DEPARTMENT CATEGORIES")
print("=" * 70)
print(df["Department"].unique())


# ============================================================
# STEP 9: DETECT INVALID AGE VALUES
# ============================================================

# Assume that the valid employee age range is 18 to 65 years

invalid_ages = df[
    (df["Age"] < 18) |
    (df["Age"] > 65)
]

print("\nINVALID AGE VALUES")
print("=" * 70)
print(invalid_ages[["Employee_ID", "Age"]])


# Convert invalid ages into missing values
df.loc[
    (df["Age"] < 18) |
    (df["Age"] > 65),
    "Age"
] = np.nan


# ============================================================
# STEP 10: MISSING-VALUE ANALYSIS
# ============================================================

missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100


missing_summary = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_percentage
})


print("\nMISSING-VALUE SUMMARY")
print("=" * 70)
print(missing_summary)


# ============================================================
# STEP 11: UNIVARIATE EDA
# ============================================================

# ---------------------------
# 11.1 Age distribution
# ---------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    bins=10,
    kde=True,
    color="steelblue"
)

plt.title("Distribution of Employee Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ---------------------------
# 11.2 Salary distribution
# ---------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Salary",
    bins=10,
    kde=True,
    color="green"
)

plt.title("Distribution of Salary")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ---------------------------
# 11.3 Salary box plot
# ---------------------------

plt.figure(figsize=(8, 4))

sns.boxplot(
    data=df,
    x="Salary",
    color="orange"
)

plt.title("Box Plot of Salary")
plt.xlabel("Salary")
plt.tight_layout()
plt.show()


# ---------------------------
# 11.4 Department frequency
# ---------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Department",
    order=df["Department"].value_counts().index,
    color="cornflowerblue"
)

plt.title("Number of Employees in Each Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()


# ---------------------------
# 11.5 Gender frequency
# ---------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Gender",
    color="mediumpurple"
)

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()


# ---------------------------
# 11.6 Target-class distribution
# ---------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Purchased_Course",
    color="salmon"
)

plt.title("Course Purchase Distribution")
plt.xlabel("Purchased Course")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 12: NUMERICAL SUMMARY
# ============================================================

print("\nNUMERICAL FEATURE SUMMARY")
print("=" * 70)

numerical_summary = df[
    ["Age", "Experience", "Salary"]
].agg([
    "count",
    "mean",
    "median",
    "std",
    "min",
    "max"
])

print(numerical_summary)


print("\nSALARY SKEWNESS")
print("=" * 70)
print(df["Salary"].skew())


print("\nDEPARTMENT FREQUENCY")
print("=" * 70)
print(df["Department"].value_counts())


print("\nTARGET PERCENTAGE")
print("=" * 70)

target_percentage = (
    df["Purchased_Course"]
    .value_counts(normalize=True)
    .mul(100)
)

print(target_percentage)


# ============================================================
# STEP 13: BIVARIATE EDA
# ============================================================

# ---------------------------
# 13.1 Experience versus salary
# ---------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Experience",
    y="Salary",
    hue="Purchased_Course",
    s=100
)

plt.title("Experience versus Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.tight_layout()
plt.show()


# ---------------------------
# 13.2 Salary by department
# ---------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Department",
    y="Salary"
)

plt.title("Salary Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Salary")
plt.tight_layout()
plt.show()


# ---------------------------
# 13.3 Average salary by department
# ---------------------------

average_salary = df.groupby(
    "Department",
    dropna=False
)["Salary"].mean()

print("\nAVERAGE SALARY BY DEPARTMENT")
print("=" * 70)
print(average_salary)


# ---------------------------
# 13.4 Course purchase by department
# ---------------------------

purchase_table = pd.crosstab(
    df["Department"],
    df["Purchased_Course"]
)

print("\nCOURSE PURCHASE BY DEPARTMENT")
print("=" * 70)
print(purchase_table)


plt.figure(figsize=(8, 5))


sns.countplot(
    data=df,
    x="Department",
    hue="Purchased_Course"
)

plt.title("Course Purchase by Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 14: MULTIVARIATE EDA
# ============================================================

# Correlation matrix
correlation_matrix = df[
    ["Age", "Experience", "Salary"]
].corr()

print("\nCORRELATION MATRIX")
print("=" * 70)
print(correlation_matrix)


# Correlation heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Pair plot
pairplot_data = df[
    [
        "Age",
        "Experience",
        "Salary",
        "Purchased_Course"
    ]
].dropna()

sns.pairplot(
    pairplot_data,
    hue="Purchased_Course",
    diag_kind="hist"
)

plt.show()


# ============================================================
# STEP 15: DETECT SALARY OUTLIERS USING IQR
# ============================================================

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


print("\nIQR-BASED SALARY OUTLIER DETECTION")
print("=" * 70)

print("First quartile Q1:", Q1)
print("Third quartile Q3:", Q3)
print("Interquartile range:", IQR)
print("Lower boundary:", lower_bound)
print("Upper boundary:", upper_bound)


salary_outliers = df[
    (df["Salary"] < lower_bound) |
    (df["Salary"] > upper_bound)
]

print("\nDETECTED SALARY OUTLIERS")
print("=" * 70)

print(
    salary_outliers[
        [
            "Employee_ID",
            "Salary",
            "Department"
        ]
    ]
)


# ============================================================
# STEP 16: SEPARATE INPUT FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Age",
        "Gender",
        "Department",
        "Experience",
        "Salary"
    ]
].copy()


# Convert target into numerical form
y = df["Purchased_Course"].map({
    "No": 0,
    "Yes": 1
})


print("\nINPUT FEATURES")
print("=" * 70)
print(X.head())


print("\nTARGET VALUES")
print("=" * 70)
print(y.head())


# ============================================================
# STEP 17: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


print("\nTRAINING AND TESTING SHAPES")
print("=" * 70)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# Create independent copies
X_train = X_train.copy()
X_test = X_test.copy()


# ============================================================
# STEP 18: CALCULATE IQR BOUNDARIES FROM TRAINING DATA ONLY
# ============================================================

# This avoids data leakage.

train_Q1 = X_train["Salary"].quantile(0.25)
train_Q3 = X_train["Salary"].quantile(0.75)

train_IQR = train_Q3 - train_Q1

train_lower_bound = train_Q1 - 1.5 * train_IQR
train_upper_bound = train_Q3 + 1.5 * train_IQR


print("\nTRAINING-DATA SALARY BOUNDARIES")
print("=" * 70)

print("Training Q1:", train_Q1)
print("Training Q3:", train_Q3)
print("Training IQR:", train_IQR)
print("Training lower boundary:", train_lower_bound)
print("Training upper boundary:", train_upper_bound)


# ============================================================
# STEP 19: CAP SALARY OUTLIERS
# ============================================================

# Use boundaries obtained only from training data.

X_train["Salary"] = X_train["Salary"].clip(
    lower=train_lower_bound,
    upper=train_upper_bound
)

X_test["Salary"] = X_test["Salary"].clip(
    lower=train_lower_bound,
    upper=train_upper_bound
)


# ============================================================
# STEP 20: DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = [
    "Age",
    "Experience",
    "Salary"
]

categorical_features = [
    "Gender",
    "Department"
]


# ============================================================
# STEP 21: NUMERICAL PREPROCESSING PIPELINE
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "median_imputation",
            SimpleImputer(strategy="median")
        ),
        (
            "standardization",
            StandardScaler()
        )
    ]
)


# ============================================================
# STEP 22: CATEGORICAL PREPROCESSING PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "mode_imputation",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "one_hot_encoding",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# STEP 23: COMBINE BOTH PREPROCESSING PIPELINES
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical_processing",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical_processing",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# STEP 24: CREATE COMPLETE MACHINE-LEARNING PIPELINE
# ============================================================

complete_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# STEP 25: TRAIN THE MODEL
# ============================================================

complete_model.fit(
    X_train,
    y_train
)

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY")


# ============================================================
# STEP 26: MAKE PREDICTIONS
# ============================================================

y_pred = complete_model.predict(X_test)

y_probability = complete_model.predict_proba(
    X_test
)[:, 1]


prediction_results = pd.DataFrame({
    "Actual_Value": y_test.to_numpy(),
    "Predicted_Value": y_pred,
    "Probability_of_Purchase": y_probability
})


print("\nPREDICTION RESULTS")
print("=" * 70)
print(prediction_results)


# ============================================================
# STEP 27: EVALUATE THE MODEL
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

confusion = confusion_matrix(
    y_test,
    y_pred
)

report = classification_report(
    y_test,
    y_pred,
    target_names=["No", "Yes"],
    zero_division=0
)


print("\nMODEL ACCURACY")
print("=" * 70)
print(accuracy)


print("\nCONFUSION MATRIX")
print("=" * 70)
print(confusion)


print("\nCLASSIFICATION REPORT")
print("=" * 70)
print(report)


# ============================================================
# STEP 28: VISUALIZE THE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    confusion,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 29: DISPLAY FINAL CONCLUSION
# ============================================================

print("\nFINAL CONCLUSION")
print("=" * 70)

print("""
The dataset was inspected and cleaned successfully.
Duplicate employee records were removed.
Inconsistent categorical values were standardized.
Invalid age values were converted into missing values.
Salary outliers were detected using the IQR method.
Training-data boundaries were used to cap salary outliers.
Missing numerical and categorical values were imputed.
Categorical features were one-hot encoded.
Numerical features were standardized.
A Logistic Regression model was trained and evaluated.
""")