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

# 2. Check for missing values
print("\nChecking for missing values...")
print(df.isnull().sum())

# 3. Define features and targets
X = df.drop(['Future_Units', 'Future_Bill'], axis=1)
y_units = df['Future_Units']
y_bill = df['Future_Bill']

# 4. Train-test split
print("\nSplitting data...")
X_train, X_test, y_units_train, y_units_test = train_test_split(X, y_units, test_size=0.2, random_state=42)
_, _, y_bill_train, y_bill_test = train_test_split(X, y_bill, test_size=0.2, random_state=42)

# 5. Initialize models
print("\nTraining models...")
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression()
}

# 6. Train and evaluate models for Future Units
print("\n--- Future Units Consumed Prediction ---")
unit_results = {}
for name, model in models.items():
    model.fit(X_train, y_units_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_units_test, pred)
    r2 = r2_score(y_units_test, pred)
    unit_results[name] = (model, mae, r2)
    print(f"{name} MAE: {mae}")
    print(f"{name} R2 Score: {r2}")

# Select best model by R2 score
best_units_model_name = max(unit_results.items(), key=lambda x: x[1][2])[0]
best_units_model = unit_results[best_units_model_name][0]
print(f"\nSelected Best Units Model: {best_units_model_name}")

# 7. Train and evaluate models for Future Bill
print("\n--- Future Bill Prediction ---")
bill_results = {}
for name, model in models.items():
    model.fit(X_train, y_bill_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_bill_test, pred)
    r2 = r2_score(y_bill_test, pred)
    bill_results[name] = (model, mae, r2)
    print(f"{name} MAE: {mae}")
    print(f"{name} R2 Score: {r2}")

# Select best model by R2 score
best_bill_model_name = max(bill_results.items(), key=lambda x: x[1][2])[0]
best_bill_model = bill_results[best_bill_model_name][0]
print(f"\nSelected Best Bill Model: {best_bill_model_name}")

# 8. Save the best models
print("\nSaving the best models...")
with open('units_model.pkl', 'wb') as f:
    pickle.dump(best_units_model, f)

with open('bill_model.pkl', 'wb') as f:
    pickle.dump(best_bill_model, f)

print("\n✅ Training Completed Successfully!")
print("Best Units Model saved as 'units_model.pkl'")    
print("Best Bill Model saved as 'bill_model.pkl'")
