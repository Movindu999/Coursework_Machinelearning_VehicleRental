from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(title="Vehicle Rental Revenue Prediction API", version="2.1")

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # coursework
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent  # /backend

# =========================
# ✅ EXISTING MODEL (Daily Revenue)
# =========================
MODEL_PATH = BASE_DIR.parent / "models" / "best_revenue_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle.get("features", ["Year", "Month", "IsSeason", "IsWeekend"])
SEASON_MONTHS = set(bundle.get("season_months", [10, 11, 12, 1, 2, 3]))
METRICS = bundle.get("metrics", {})

def make_features_from_date(date_str: str) -> pd.DataFrame:
    d = pd.to_datetime(date_str)
    year = int(d.year)
    month = int(d.month)
    is_weekend = 1 if d.weekday() >= 5 else 0
    is_season = 1 if month in SEASON_MONTHS else 0
    return pd.DataFrame([[year, month, is_season, is_weekend]], columns=FEATURES)


# =========================
# ✅ EXISTING MODEL (Vehicle + Date Revenue)
# =========================
VEHICLE_MODEL_PATH = BASE_DIR.parent / "models" / "best_vehicle_revenue_model.pkl"

vehicle_model = None
vehicle_features = None
vehicle_le = None
vehicle_metrics = {}

if VEHICLE_MODEL_PATH.exists():
    vehicle_bundle = joblib.load(VEHICLE_MODEL_PATH)
    vehicle_model = vehicle_bundle["model"]
    vehicle_features = vehicle_bundle.get(
        "features",
        ["Year", "Month", "IsSeason", "IsWeekend", "VehicleEncoded"]
    )
    vehicle_le = vehicle_bundle.get("label_encoder", None)
    vehicle_metrics = vehicle_bundle.get("metrics", {})
else:
    print(f"⚠ Vehicle model not found: {VEHICLE_MODEL_PATH}")


# =========================
# ✅ NEW MODEL (Demand / Bookings Count)
# =========================
DEMAND_MODEL_PATH = BASE_DIR.parent / "models" / "best_demand_model.pkl"

demand_model = None
demand_features = None
demand_vehicle_types = ["Car", "Bike", "Tuk Tuk"]
demand_metrics = {}
DEMAND_SEASON_MONTHS = [10, 11, 12, 1, 2, 3]

if DEMAND_MODEL_PATH.exists():
    demand_bundle = joblib.load(DEMAND_MODEL_PATH)
    demand_model = demand_bundle.get("model", None)
    demand_features = demand_bundle.get("features", None)
    demand_vehicle_types = demand_bundle.get("vehicle_types", demand_vehicle_types)
    demand_metrics = demand_bundle.get("metrics", {})
    DEMAND_SEASON_MONTHS = demand_bundle.get("season_months", DEMAND_SEASON_MONTHS)
else:
    print(f"⚠ Demand model not found: {DEMAND_MODEL_PATH}")

DEMAND_SEASON_MONTHS_SET = set(DEMAND_SEASON_MONTHS)

def make_demand_features(date_str: str, vehicle_type: str) -> pd.DataFrame:
    """
    Build X with the exact columns in demand_features (saved from training).
    Expected columns include:
      Year, Month, DayOfWeek, IsSeason, IsWeekend, VehicleType_<type>...
    """
    if demand_features is None:
        raise ValueError("Demand model features not available (model not loaded).")

    d = pd.to_datetime(date_str)
    year = int(d.year)
    month = int(d.month)
    day_of_week = int(d.weekday())
    is_weekend = 1 if day_of_week >= 5 else 0
    is_season = 1 if month in DEMAND_SEASON_MONTHS_SET else 0

    row = {col: 0 for col in demand_features}

    # Set numeric time features if present
    if "Year" in row: row["Year"] = year
    if "Month" in row: row["Month"] = month
    if "DayOfWeek" in row: row["DayOfWeek"] = day_of_week
    if "IsSeason" in row: row["IsSeason"] = is_season
    if "IsWeekend" in row: row["IsWeekend"] = is_weekend

    # One-hot for vehicle type (pd.get_dummies keeps spaces)
    dummy_col = f"VehicleType_{vehicle_type}"
    if dummy_col not in row:
        # try soft match by stripping spaces (just in case)
        possible = [c for c in row.keys() if c.replace(" ", "").lower() == dummy_col.replace(" ", "").lower()]
        if possible:
            dummy_col = possible[0]
        else:
            raise ValueError(
                f"Vehicle type '{vehicle_type}' not supported by demand model. "
                f"Expected one of: {demand_vehicle_types}"
            )
    row[dummy_col] = 1

    return pd.DataFrame([[row[c] for c in demand_features]], columns=demand_features)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models/info")
def models_info():
    # Keep existing output unchanged + add demand model info safely
    return {
        "features": FEATURES,
        "season_months": sorted(list(SEASON_MONTHS)),
        "metrics": METRICS,

        "vehicle_model_loaded": vehicle_model is not None,
        "vehicle_model_metrics": vehicle_metrics,

        # ✅ NEW
        "demand_model_loaded": demand_model is not None,
        "demand_model_metrics": demand_metrics,
        "demand_vehicle_types": demand_vehicle_types
    }


@app.get("/predict/revenue")
def predict_revenue(date: str = Query(..., description="Format: YYYY-MM-DD")):
    try:
        X = make_features_from_date(date)
        pred = float(model.predict(X)[0])
        month = int(pd.to_datetime(date).month)

        return {
            "date": date,
            "predicted_revenue_lkr": round(pred, 2),
            "is_season": 1 if month in SEASON_MONTHS else 0,
            "is_weekend": int(pd.to_datetime(date).weekday() >= 5)
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/vehicles")
def get_vehicle_types():
    if vehicle_le is not None:
        return {"vehicle_types": list(vehicle_le.classes_)}
    return {"vehicle_types": ["Car", "Bike", "Tuk Tuk"]}


@app.get("/predict/vehicle-revenue")
def predict_vehicle_revenue(
    date: str = Query(..., description="Format: YYYY-MM-DD"),
    vehicle_type: str = Query(..., description="Must match dataset vehicle types exactly")
):
    try:
        d = pd.to_datetime(date)
        year = int(d.year)
        month = int(d.month)
        is_weekend = 1 if d.weekday() >= 5 else 0
        is_season = 1 if month in SEASON_MONTHS else 0

        if vehicle_model is not None and vehicle_le is not None:
            if vehicle_type not in list(vehicle_le.classes_):
                return {"error": f"Invalid vehicle_type: {vehicle_type}. Use /vehicles to see valid values."}

            vehicle_encoded = int(vehicle_le.transform([vehicle_type])[0])

            X = pd.DataFrame(
                [[year, month, is_season, is_weekend, vehicle_encoded]],
                columns=vehicle_features
            )

            pred = float(vehicle_model.predict(X)[0])
        else:
            X = make_features_from_date(date)
            base_pred = float(model.predict(X)[0])

            multipliers = {"Car": 1.2, "Bike": 0.6, "Tuk Tuk": 0.8}
            multiplier = multipliers.get(vehicle_type, 1.0)
            pred = base_pred * multiplier

        return {
            "date": date,
            "vehicle_type": vehicle_type,
            "predicted_vehicle_revenue_lkr": round(pred, 2),
            "is_season": is_season,
            "is_weekend": is_weekend
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# ✅ NEW: Demand prediction (date + vehicle)
# =========================
@app.get("/predict/demand")
def predict_demand(
    date: str = Query(..., description="Format: YYYY-MM-DD"),
    vehicle_type: str = Query(..., description="Car/Bike/Tuk Tuk")
):
    try:
        if demand_model is None:
            return {"error": f"Demand model not loaded. Train & save: {DEMAND_MODEL_PATH.name}"}

        X = make_demand_features(date, vehicle_type)
        pred = float(demand_model.predict(X)[0])

        # bookings count should not be negative
        pred = max(0.0, pred)

        d = pd.to_datetime(date)
        month = int(d.month)
        is_weekend = int(d.weekday() >= 5)
        is_season = 1 if month in DEMAND_SEASON_MONTHS_SET else 0

        return {
            "date": date,
            "vehicle_type": vehicle_type,
            "predicted_bookings": round(pred, 2),
            "is_season": is_season,
            "is_weekend": is_weekend
        }
    except Exception as e:
        return {"error": str(e)}


# =========================
# ✅ NEW: Demand-based recommendation (date only)
# =========================
@app.get("/recommend/demand")
def recommend_by_demand(
    date: str = Query(..., description="Format: YYYY-MM-DD")
):
    try:
        if demand_model is None:
            return {"error": f"Demand model not loaded. Train & save: {DEMAND_MODEL_PATH.name}"}

        preds = []
        for vt in demand_vehicle_types:
            X = make_demand_features(date, vt)
            p = float(demand_model.predict(X)[0])
            preds.append({"vehicle_type": vt, "predicted_bookings": max(0.0, p)})

        # pick best
        best = max(preds, key=lambda x: x["predicted_bookings"])
        recommended = best["vehicle_type"]

        d = pd.to_datetime(date)
        month = int(d.month)
        is_weekend = int(d.weekday() >= 5)
        is_season = 1 if month in DEMAND_SEASON_MONTHS_SET else 0

        return {
            "date": date,
            "recommended_vehicle": recommended,
            "predictions": [
                {"vehicle_type": x["vehicle_type"], "predicted_bookings": round(x["predicted_bookings"], 2)}
                for x in preds
            ],
            "is_season": is_season,
            "is_weekend": is_weekend
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)