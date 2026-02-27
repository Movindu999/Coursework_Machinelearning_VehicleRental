import joblib
import pandas as pd

# Load the revenue model
bundle = joblib.load('models/best_revenue_model.pkl')

print("=" * 60)
print("REVENUE MODEL INSPECTION")
print("=" * 60)
print(f"Model type: {type(bundle['model'])}")
print(f"Features in bundle: {bundle.get('features')}")

if hasattr(bundle['model'], 'n_features_in_'):
    print(f"Model expects {bundle['model'].n_features_in_} features")
    
if hasattr(bundle['model'], 'feature_names_in_'):
    print(f"Feature names in model: {list(bundle['model'].feature_names_in_)}")

# Test prediction
print("\n" + "=" * 60)
print("TESTING PREDICTION")
print("=" * 60)
try:
    test_data = pd.DataFrame([[2026, 2, 1, 0]], columns=bundle.get('features', ['Year', 'Month', 'IsSeason', 'IsWeekend']))
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data columns: {list(test_data.columns)}")
    pred = bundle['model'].predict(test_data)
    print(f"Prediction successful: {pred[0]}")
except Exception as e:
    print(f"ERROR: {e}")
