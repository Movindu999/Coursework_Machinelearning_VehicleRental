🚗 Vehicle Rental ML Dashboard

📁 Project Structure
vehicle-rental-ml-dashboard/
│
├── backend/
│   ├── app.py
│   └── models/
│       ├── best_revenue_model.pkl
│       ├── best_vehicle_revenue_model.pkl
│       └── best_demand_model.pkl
│
├── frontend/
│   ├── index.html
│   └── style.css
│
├── notebooks/
│   ├── revenue_training.ipynb
│   └── demand_training.ipynb
│
├── dataset/
│   └── rentals_transactions_realistic.csv
│
└── README.md


A Machine Learning powered web dashboard for predicting rental revenue and vehicle demand in a vehicle rental business.

This project predicts:

📅 Daily Total Revenue

🚘 Vehicle-Specific Revenue

📊 Vehicle Comparison (Car / Bike / Tuk Tuk)

🔥 Demand-Based Vehicle Recommendation

Built using Python, FastAPI, Scikit-Learn, HTML, CSS, JavaScript, and Chart.js.

📌 Project Overview

Vehicle rental businesses experience variations in revenue and demand depending on:

Season (Oct–Mar vs Apr–Sep)

Weekday vs Weekend

Vehicle Type (Car, Bike, Tuk Tuk)

This system uses historical transaction data and Machine Learning models to:

Forecast future revenue

Compare vehicle performance

Recommend the best vehicle based on predicted demand

🧠 Machine Learning Models Used

The following regression models were trained and evaluated:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

K-Nearest Neighbors (KNN) Regressor

Support Vector Regressor (SVR)

Model Evaluation Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

5-Fold Cross Validation

Best models were selected using performance metrics and saved as .pkl files.

📊 Features
1️⃣ Daily Revenue Prediction

Predicts total expected revenue (LKR) for a selected date.

2️⃣ Vehicle Revenue Prediction

Predicts revenue for a selected vehicle type (Car / Bike / Tuk Tuk) on a given date.

3️⃣ Vehicle Comparison Chart

Compares predicted revenue of all vehicle types for the same date.

4️⃣ Demand-Based Recommendation (Professional Feature)

Predicts bookings count for each vehicle type and recommends the vehicle with the highest predicted demand.

This is different from revenue comparison and provides better operational insight.

🏗️ System Architecture
🔹 ML Training Layer

Jupyter Notebooks

Data preprocessing

Feature engineering

Model training & tuning

Model saving (.pkl files)

🔹 Backend API (FastAPI)

Handles model loading and prediction endpoints:

/predict/revenue

/predict/vehicle-revenue

/recommend/demand

/vehicles

/health

🔹 Frontend Dashboard

HTML + CSS

JavaScript (Fetch API)

Chart.js for visualizations


⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/vehicle-rental-ml-dashboard.git
cd vehicle-rental-ml-dashboard
2️⃣ Install Dependencies
pip install -r requirements.txt

If requirements.txt is not available:

pip install fastapi uvicorn pandas scikit-learn joblib
🚀 Run Backend
uvicorn app:app --host 0.0.0.0 --port 8003

Backend will run at:

http://127.0.0.1:8003
🌐 Run Frontend

Open index.html inside the frontend folder in your browser.

Make sure backend is running before using the dashboard.

📈 Example API Endpoints

Daily Revenue:

GET /predict/revenue?date=2026-05-21

Vehicle Revenue:

GET /predict/vehicle-revenue?date=2026-05-21&vehicle_type=Car

Demand Recommendation:

GET /recommend/demand?date=2026-05-21
📌 Key Highlights

Uses 5 ML algorithms

80/20 Train-Test split

Cross Validation

Hyperparameter tuning with GridSearchCV

Professional dashboard UI

Revenue-based and Demand-based insights

Fully functional REST API

🔮 Future Improvements

Add holiday and weather data

Add lag features (previous day demand)

Integrate advanced models (XGBoost / Gradient Boosting)

Add profit optimization module

Deploy on cloud (Render / Railway / AWS)

👨‍💻 Developers -

Movindu Lithmin , Harindu Harshith , Dilki Himasha , Dilki Chamoda
Machine Learning Coursework Project
