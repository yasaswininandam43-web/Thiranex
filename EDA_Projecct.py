# Predictive Modeling Using Machine Learning

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset (corrected filename casing to match workspace)
data = pd.read_csv("StudentsPerformance.csv")

# Display the first five rows
print("First five rows of the dataset:")
print(data.head())

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Remove rows with missing values
data = data.dropna()

# Select input and output columns based on actual dataset columns
# We will predict 'math score' using 'reading score' and 'writing score'
X = data[["reading score", "writing score"]]
y = data["math score"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the machine learning model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict the test data
predictions = model.predict(X_test)

# Check model performance
mae = mean_absolute_error(y_test, predictions)
accuracy_score = r2_score(y_test, predictions)

print("\nMean Absolute Error:", round(mae, 2))
print("R2 Score:", round(accuracy_score, 2))

# Compare actual and predicted values
comparison = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": predictions.round(2)
})

print("\nActual and Predicted Scores:")
print(comparison.head(10))

# Visualize actual and predicted scores
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("Actual vs Predicted Math Performance")
plt.show()
