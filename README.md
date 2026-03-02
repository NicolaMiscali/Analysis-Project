# 🚕 NYC Yellow Taxi Trip Duration — End-to-End ML Project

A complete data science project built on the [NYC TLC Yellow Taxi Trip Records](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) dataset from Kaggle. The goal is to **predict taxi trip duration** in New York City using regression models, combined with thorough data exploration and a PostgreSQL-backed pipeline.

---

## 📁 Project Structure
```
nyc-taxi/
├── db_config.ipynb      # Database setup, ETL pipeline, data cleaning
├── taxy_eda.ipynb       # Exploratory Data Analysis (EDA)
├── taxi_ml.ipynb        # Machine Learning — model training and evaluation
└── README.md
```

---

## 📦 Dataset

| Property | Value |
|---|---|
| Source | Kaggle — NYC Taxi Trip Duration |
| Raw rows | 1,458,644 |
| Columns | 11 |
| Time range | January 2016 – June 2016 |
| Target variable | `trip_duration` (seconds) |

**Original features:** `id`, `vendor_id`, `pickup_datetime`, `dropoff_datetime`, `passenger_count`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `store_and_fwd_flag`, `trip_duration`

---

## 🗄️ Phase 1 — Database Setup & ETL (`db_config.ipynb`)

### Database
- Loaded raw CSV into **PostgreSQL** (`taxi_nyc_db`) via **SQLAlchemy** + `psycopg2`
- Raw data stored in table `taxi_nyc`
- Cleaned data stored in table `taxi_nyc_clean`

### Data Quality Check
- **Null values:** 0 across all columns
- **Duplicates:** 0

### Feature Engineering

| Feature | Description |
|---|---|
| `pickup_hour` | Hour of pickup (0–23) |
| `pickup_day` | Day of month |
| `pickup_month` | Month (1–6) |
| `pickup_weekday` | Day of week (0=Monday, 6=Sunday) |
| `is_weekend` | Binary flag (1 = Saturday or Sunday) |
| `day_of_the_week` | Weekday name |
| `month_name` | Month name |
| `duration_trip_calc` | Cross-validated duration (dropoff − pickup in seconds) |
| `distance_km` | Haversine distance between pickup and dropoff |
| `Speed` | distance_km / (trip_duration / 3600) |
| `trip_duration_min` | trip_duration converted to minutes |

### Distance — Haversine Formula
The dataset has no distance column. Distance was computed from coordinates:
```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))
```

> Average speed after calculation: **~14.4 km/h** — consistent with NYC's heavy traffic. Studies confirm cycling in Manhattan is often faster than a taxi at peak hours.

### Data Cleaning
Outliers removed using domain knowledge thresholds:
```python
df_clean = df[
    (df['Speed'] > 0)          & (df['Speed'] < 100)          &
    (df['trip_duration'] > 60) & (df['trip_duration'] < 7200) &
    (df['distance_km'] > 0.1)  & (df['distance_km'] < 100)
]
```

| | Raw | Clean |
|---|---|---|
| Rows | 1,458,644 | 1,440,078 |
| Removed | — | 18,566 (~1.3%) |
| Max trip duration | ~40 days (error) | ~2 hours |

---

## 📊 Phase 2 — Exploratory Data Analysis (`taxy_eda.ipynb`)

### Key Statistics

| Metric | Value |
|---|---|
| Average trip duration | **14.0 min** |
| Average distance | **3.47 km** |
| Average speed | **14.5 km/h** |
| Solo rides (1 passenger) | **70.8%** |
| Dominant vendor | Vendor 2 (53.4%) |

### Visualizations
- Passenger count distribution — pie chart
- Monthly trip distribution — pie chart (Jan–Jun, ~17% each)
- Box plots — speed, distance, trip duration side by side
- Histograms — speed, distance, trip duration with mean lines
- NYC geographic scatter plot — pickup coordinates reveal Manhattan's shape
- Vendor vs trip duration — boxplot
- Correlation heatmap — `distance_km` is the dominant predictor

---

## 🤖 Phase 3 — Machine Learning (`taxi_ml.ipynb`)

### Setup
- **300,000 rows** sampled for iteration speed (`random_state=42`)
- Train/Test split: **80% / 20%**
- Target variable: `trip_duration` (seconds)

### Features
```python
features = [
    'passenger_count', 'distance_km',
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude',
    'is_weekend', 'pickup_hour',
    'pickup_month', 'pickup_weekday', 'vendor_id'
]
```

> `Speed`, `duration_trip_calc` and `trip_duration_min` were excluded to prevent **data leakage** — they are derived from the target variable.

### Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 213.33 sec | 323.31 sec | 0.7557 |
| LightGBM (300k) | 198.12 sec | 303.24 sec | 0.7851 |
| **LightGBM (full 1.4M)** | **197.22 sec** | **302.51 sec** | **0.7852** ✅ |

> The near-identical performance between 300k and 1.4M rows shows the model was not data-limited. The next step is hyperparameter tuning.

### Feature Importance (LightGBM)
`distance_km` is the dominant predictor, followed by `pickup_hour` and location coordinates. `passenger_count` and `vendor_id` carry near-zero importance.

### Evaluation Plots
- Feature importance bar chart
- Predicted vs Actual scatter plot
- Residuals vs Predicted scatter plot
- Residuals distribution histogram

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| SQLAlchemy + psycopg2 | PostgreSQL connection |
| PostgreSQL | Data storage |
| scikit-learn | ML pipeline and metrics |
| LightGBM | Final regression model |
| Jupyter Notebook | Development environment |

---

## ⚙️ Setup
```bash
pip install pandas numpy matplotlib seaborn sqlalchemy psycopg2-binary lightgbm scikit-learn jupyter
```

Update the DB credentials in `db_config.ipynb`:
```python
DB_USER     = 'your_user'
DB_PASSWORD = 'your_password'
DB_HOST     = 'localhost'
DB_PORT     = '5432'
DB_NAME     = 'taxi_nyc_db'
```

Run notebooks in order:
```
1. db_config.ipynb   →  ETL + cleaning + feature engineering
2. taxy_eda.ipynb    →  EDA + visualizations
3. taxi_ml.ipynb     →  model training + evaluation
```


## 📄 License
Data from the NYC TLC and Kaggle NYC Taxi Trip Duration competition.
