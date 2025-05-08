# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('energy_consumption_data.csv')  # Your dataset file

print("\nEncoding categorical columns...")
df['House_Size'] = df['House_Size'].map({'small': 0, 'medium': 1, 'large': 2})
df['Weather'] = df['Weather'].map({'cold': 0, 'moderate': 1, 'hot': 2})
df['Heavy_Appliances'] = df['Heavy_Appliances'].map({'few': 0, 'many': 1})

df['Heavy_Appliances'] = df['Heavy_Appliances'].fillna(df['Heavy_Appliances'].mode()[0])

# 2. Check if there are any missing values
print("\nChecking for missing values...")
print(df.isnull().sum())

# 3. Define features (X) and targets (y)
X = df.drop(['Future_Units', 'Future_Bill'], axis=1)
y_units = df['Future_Units']
y_bill = df['Future_Bill']

# 4. Split the data into training and testing sets
print("\nSplitting data...")
X_train, X_test, y_units_train, y_units_test = train_test_split(X, y_units, test_size=0.2, random_state=42)
_, _, y_bill_train, y_bill_test = train_test_split(X, y_bill, test_size=0.2, random_state=42)

# 5. Initialize models
print("\nTraining models...")
rf_units_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_units_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
lr_units_model = LinearRegression()

# 6. Train models for predicting Future Units
rf_units_model.fit(X_train, y_units_train)
gb_units_model.fit(X_train, y_units_train)
lr_units_model.fit(X_train, y_units_train)

# 7. Evaluate models for Units prediction
rf_units_pred = rf_units_model.predict(X_test)
gb_units_pred = gb_units_model.predict(X_test)
lr_units_pred = lr_units_model.predict(X_test)

print("\n--- Future Units Consumed Prediction ---")
print("Random Forest MAE:", mean_absolute_error(y_units_test, rf_units_pred))
print("Random Forest R2 Score:", r2_score(y_units_test, rf_units_pred))

print("Gradient Boosting MAE:", mean_absolute_error(y_units_test, gb_units_pred))
print("Gradient Boosting R2 Score:", r2_score(y_units_test, gb_units_pred))

print("Linear Regression MAE:", mean_absolute_error(y_units_test, lr_units_pred))
print("Linear Regression R2 Score:", r2_score(y_units_test, lr_units_pred))

best_units_model = rf_units_model

# 8. Now Train models for predicting Future Bill
rf_bill_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_bill_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
lr_bill_model = LinearRegression()

rf_bill_model.fit(X_train, y_bill_train)
gb_bill_model.fit(X_train, y_bill_train)
lr_bill_model.fit(X_train, y_bill_train)

# Predict
rf_bill_pred = rf_bill_model.predict(X_test)
gb_bill_pred = gb_bill_model.predict(X_test)
lr_bill_pred = lr_bill_model.predict(X_test)

print("\n--- Future Bill Prediction ---")
print("Random Forest MAE:", mean_absolute_error(y_bill_test, rf_bill_pred))
print("Random Forest R2 Score:", r2_score(y_bill_test, rf_bill_pred))

print("Gradient Boosting MAE:", mean_absolute_error(y_bill_test, gb_bill_pred))
print("Gradient Boosting R2 Score:", r2_score(y_bill_test, gb_bill_pred))

print("Linear Regression MAE:", mean_absolute_error(y_bill_test, lr_bill_pred))
print("Linear Regression R2 Score:", r2_score(y_bill_test, lr_bill_pred))

# Choose best model (Here RandomForest again)
best_bill_model = rf_bill_model

# 9. Save the trained models
print("\nSaving the best models...")
with open('units_model.pkl', 'wb') as f:
    pickle.dump(best_units_model, f)

with open('bill_model.pkl', 'wb') as f:
    pickle.dump(best_bill_model, f)

print("\nTraining Completed Successfully! Models are saved as 'units_model.pkl' and 'bill_model.pkl'. 🎯")
