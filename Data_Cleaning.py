#Import the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. CREATE A SAMPLE RAW DATASET


np.random.seed(42)

data = pd.DataFrame({
    "Customer_ID": range(1, 21),
    "Age": np.random.randint(18, 60, 20),
    "Gender": np.random.choice(["Male", "Female"], 20),
    "Product": np.random.choice(
        ["Laptop", "Mobile", "Tablet", "Headphones"], 20
    ),
    "Quantity": np.random.randint(1, 6, 20),
    "Price": np.random.randint(500, 50000, 20)
})

# Add missing values, an outlier, and a duplicate
data.loc[2, "Age"] = np.nan
data.loc[5, "Gender"] = np.nan
data.loc[7, "Price"] = np.nan
data.loc[10, "Quantity"] = np.nan
data.loc[15, "Price"] = 500000
data = pd.concat([data, data.iloc[[3]]], ignore_index=True)

print("========== RAW DATASET ==========")
print(data.head(10))

print("\nDataset shape:", data.shape)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:", data.duplicated().sum())


# 2. VISUALIZE RAW DATA


plt.figure(figsize=(8, 4))
sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Before Cleaning")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.boxplot(x=data["Price"])
plt.title("Price Outliers Before Cleaning")
plt.tight_layout()
plt.show()


# 3. REMOVE DUPLICATE ROWS


cleaned_data = data.drop_duplicates().copy()

print("\nDuplicates removed:", len(data) - len(cleaned_data))

# 4. HANDLE MISSING VALUES


# Fill numerical columns using their median
cleaned_data["Age"] = cleaned_data["Age"].fillna(
    cleaned_data["Age"].median()
)

cleaned_data["Quantity"] = cleaned_data["Quantity"].fillna(
    cleaned_data["Quantity"].median()
)

cleaned_data["Price"] = cleaned_data["Price"].fillna(
    cleaned_data["Price"].median()
)

# Fill categorical values using the most frequent category
cleaned_data["Gender"] = cleaned_data["Gender"].fillna(
    cleaned_data["Gender"].mode()[0]
)

print("\nMissing values after filling:")
print(cleaned_data.isnull().sum())

# 5. HANDLE PRICE OUTLIERS USING IQR CAPPING


Q1 = cleaned_data["Price"].quantile(0.25)
Q3 = cleaned_data["Price"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = max(0, Q1 - 1.5 * IQR)
upper_limit = Q3 + 1.5 * IQR

outlier_count = (
    (cleaned_data["Price"] < lower_limit) |
    (cleaned_data["Price"] > upper_limit)
).sum()

print("\nPrice outliers detected:", outlier_count)
print("Lower limit:", round(lower_limit, 2))
print("Upper limit:", round(upper_limit, 2))

# Cap extreme prices rather than deleting entire records
cleaned_data["Price"] = cleaned_data["Price"].clip(
    lower=lower_limit,
    upper=upper_limit
)


# 6. CHECK CLEANED DATA


print("\n========== CLEANED DATASET ==========")
print(cleaned_data.head(10))

print("\nFinal shape:", cleaned_data.shape)
print("\nRemaining missing values:",
      cleaned_data.isnull().sum().sum())
print("Remaining duplicate rows:",
      cleaned_data.duplicated().sum())

print("\nStatistical Summary:")
print(cleaned_data.describe())


# 7. VISUALIZE CLEANED DATA


# A. Product counts
plt.figure(figsize=(8, 5))
sns.countplot(data=cleaned_data, x="Product")
plt.title("Number of Sales by Product")
plt.xlabel("Product")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

# B. Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(cleaned_data["Price"], bins=10, kde=True)
plt.title("Distribution of Product Prices After Cleaning")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# C. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(cleaned_data["Age"], bins=10, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# D. Correlation heatmap
numeric_columns = ["Age", "Quantity", "Price"]

plt.figure(figsize=(7, 5))
sns.heatmap(
    cleaned_data[numeric_columns].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.show()

# E. Compare prices before and after cleaning
plt.figure(figsize=(9, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=data["Price"])
plt.title("Before Cleaning")

plt.subplot(1, 2, 2)
sns.boxplot(y=cleaned_data["Price"])
plt.title("After Cleaning")

plt.tight_layout()
plt.show()


# 8. CREATE A SIMPLE INSIGHTS REPORT


print("\n========== DATA INSIGHTS ==========")

print("\nProduct record counts:")
print(cleaned_data["Product"].value_counts())

print("\nAverage customer age:",
      round(cleaned_data["Age"].mean(), 2))

print("Average price:",
      round(cleaned_data["Price"].mean(), 2))

print("Total quantity:",
      int(cleaned_data["Quantity"].sum()))

print("Total recorded sales value:",
      round((cleaned_data["Price"] *
             cleaned_data["Quantity"]).sum(), 2))


# 9. SAVE THE CLEANED DATASET


cleaned_data.to_csv("cleaned_sales_data.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Download the cleaned CSV in Google Colab
from google.colab import files
files.download("cleaned_sales_data.csv")
