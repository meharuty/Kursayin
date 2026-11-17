import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)


def save_and_show(name):
    """Save the current plot into the figures folder, then show it."""
    plt.savefig("figures/" + name + ".png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------
# Load the data
# ---------------------------------------------------------------

names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
         "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_columns = ["sex", "cp", "fbs", "restecg", "exang",
                       "slope", "ca", "thal"]

data = pd.read_csv("processed.cleveland.data", header=None,
                   names=names, na_values="?")

print("Rows and columns:", data.shape)
print(data.head())

print("\nColumn types:")
print(data.dtypes)

print("\nBasic statistics:")
print(data.describe())


# ---------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------
print("\nMissing values in each column:")
print(data.isnull().sum())

rows_with_missing = data[data.isnull().any(axis=1)]
print("\nRows that have missing values:")
print(rows_with_missing)


# ---------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------
duplicates = data.duplicated().sum()
print("\nNumber of duplicate rows:", duplicates)

data = data.drop_duplicates()


# ---------------------------------------------------------------
# Check for strange values
# ---------------------------------------------------------------
allowed = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
    "target": [0, 1, 2, 3, 4],
}

print("\nChecking categorical columns:")
for column in allowed:
    found = data[column].dropna().unique()
    wrong = [v for v in found if v not in allowed[column]]
    if len(wrong) == 0:
        print(column, "- ok")
    else:
        print(column, "- strange values:", wrong)

limits = {
    "age": (18, 100),
    "trestbps": (70, 250),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0, 7),
}

print("\nChecking numeric columns:")
for column in limits:
    low, high = limits[column]
    bad = data[(data[column] < low) | (data[column] > high)]
    print(column, "- values outside", limits[column], ":", len(bad))


# ---------------------------------------------------------------
# Outliers (values far away from most of the data)
# ---------------------------------------------------------------
print("\nOutliers:")
for column in numeric_columns:
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    print(column, ":", len(outliers), "outliers")

plt.figure(figsize=(14, 4))
for i, column in enumerate(numeric_columns):
    plt.subplot(1, 5, i + 1)
    plt.boxplot(data[column].dropna())
    plt.title(column)
plt.tight_layout()
save_and_show("boxplots")


# ---------------------------------------------------------------
# Fill missing values with the median
# ---------------------------------------------------------------
print("\nMissing values before filling:")
print(data.isnull().sum())

for column in ["ca", "thal"]:
    median_value = data[column].median()
    print(column, "- median used to fill:", median_value)
    data[column] = data[column].fillna(median_value)

data = data.reset_index(drop=True)

print("\nMissing values left:", data.isnull().sum().sum())
print("Rows left:", len(data))

data["ca"] = data["ca"].astype(int)
data["thal"] = data["thal"].astype(int)


# ---------------------------------------------------------------
# Make the target binary
# ---------------------------------------------------------------
print("\nTarget before:")
print(data["target"].value_counts().sort_index())

data["target"] = (data["target"] > 0).astype(int)

print("\nTarget after (0 = healthy, 1 = sick):")
print(data["target"].value_counts())

plt.figure(figsize=(4, 3))
data["target"].value_counts().sort_index().plot(kind="bar")
plt.title("Number of healthy (0) and sick (1) patients")
plt.xticks(rotation=0)
save_and_show("class_balance")


# ---------------------------------------------------------------
# Look at the numeric features
# ---------------------------------------------------------------
print("\nSkewness of numeric features:")
print(data[numeric_columns].skew().round(2))

print("\nAverage of each numeric feature for healthy and sick patients:")
print(data.groupby("target")[numeric_columns].mean().round(2))

plt.figure(figsize=(14, 4))
for i, column in enumerate(numeric_columns):
    plt.subplot(1, 5, i + 1)
    plt.hist(data[data["target"] == 0][column], alpha=0.6, label="healthy")
    plt.hist(data[data["target"] == 1][column], alpha=0.6, label="sick")
    plt.title(column)
    plt.legend()
plt.tight_layout()
save_and_show("numeric_histograms")


# ---------------------------------------------------------------
# Look at the categorical features
# ---------------------------------------------------------------
print("\nShare of sick patients for each value of the categorical features:")
for column in categorical_columns:
    print("\n" + column)
    print(data.groupby(column)["target"].agg(["count", "mean"]).round(2))

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, column in zip(axes.ravel(), categorical_columns):
    table = pd.crosstab(data[column], data["target"])
    table.plot(kind="bar", ax=ax, rot=0)
    ax.set_title(column)
    ax.set_xlabel("")
    ax.legend(["healthy", "sick"])
plt.tight_layout()
save_and_show("categorical_features")


# ---------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------
correlation = data.corr()
print("\nCorrelation with the target:")
print(correlation["target"].sort_values(ascending=False).round(3))

plt.figure(figsize=(9, 7))
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(correlation)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation)), correlation.columns)
plt.title("Correlation matrix")
save_and_show("correlation_matrix")

feature_corr = data.drop(columns="target").corr().abs()
feature_names = list(feature_corr.columns)

print("\nStrongly correlated feature pairs (above 0.5):")
found_pair = False
for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
        value = feature_corr.iloc[i, j]
        if value > 0.5:
            print(feature_names[i], "and", feature_names[j], ":", round(value, 2))
            found_pair = True
if not found_pair:
    print("none")


# ---------------------------------------------------------------
# Split into train and test (80% / 20%)
# ---------------------------------------------------------------

train_parts = []
test_parts = []

for label in [0, 1]:
    group = data[data["target"] == label]
    group = group.sample(frac=1, random_state=42)   # shuffle
    cut = int(len(group) * 0.8)
    train_parts.append(group.iloc[:cut])
    test_parts.append(group.iloc[cut:])

train = pd.concat(train_parts).sample(frac=1, random_state=42)
test = pd.concat(test_parts).sample(frac=1, random_state=42)

print("\nTrain size:", len(train))
print("Test size:", len(test))
print("Sick share in train:", round(train["target"].mean(), 2))
print("Sick share in test:", round(test["target"].mean(), 2))


# ---------------------------------------------------------------
# Final check and save
# ---------------------------------------------------------------
assert data.isnull().sum().sum() == 0, "there are still missing values"
assert set(data["target"].unique()) == {0, 1}, "target is not binary"

data.to_csv("cleveland_clean.csv", index=False)
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("\nSaved: cleveland_clean.csv, train.csv, test.csv")
print("Plots saved in the folder: figures")
