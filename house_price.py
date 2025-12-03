# HOUSE PRICE PREDICTION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("USA_Housing.csv")
df.head()

# Features & target
X = df[["Avg Area Income", "Avg Area House Age", "Avg Area Number of Rooms",
        "Avg Area Number of Bedrooms", "Area Population"]]
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
