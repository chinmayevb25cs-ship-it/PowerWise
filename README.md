# ⚡ PowerWise — Household Electricity Consumption Forecasting

PowerWise is a machine learning project that analyzes historical household electricity usage and predicts future average power consumption. It combines data preprocessing, regression modelling, model evaluation, and an interactive Streamlit dashboard.

## 🎯 Project Objective

The goal is to explore electricity consumption patterns and evaluate whether a machine learning model can predict future power usage more accurately than a simple baseline.

## 📊 Dataset

- **Source:** UCI Individual Household Electric Power Consumption dataset
- **Original data:** Minute-level household electricity measurements
- **Processed data:** Hourly average active power
- **Target:** Average active power 24 hours ahead
- **Raw dataset size:** 2,075,259 rows and 9 columns

## 🛠️ Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit
- Joblib

## ⚙️ Project Workflow

1. Load and inspect the raw electricity dataset.
2. Handle missing power readings.
3. Aggregate minute-level readings into hourly averages.
4. Create time-based and historical lag features.
5. Train a Random Forest regression model.
6. Compare it against a previous-day baseline.
7. Evaluate predictions using MAE and RMSE.
8. Display results in an interactive dashboard.

## 📈 Initial Model Results

On the chronological test split used during the initial training run:

| Model | MAE | RMSE |
|---|---:|---:|
| Previous-day baseline | 0.5619 kW | 0.8199 kW |
| Random Forest | 0.4428 kW | 0.6056 kW |

The Random Forest reduced MAE by approximately **21.2%** and RMSE by approximately **26.1%** compared with the baseline in that run.

These are initial evaluation results, not a guarantee of future performance. The forecasting setup and test methodology should be validated further.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd PowerWise
```

Replace the placeholder with your repository URL after you publish the project.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install pandas numpy matplotlib scikit-learn streamlit plotly joblib
```

### 4. Prepare the dataset

Download the UCI Individual Household Electric Power Consumption dataset. Extract `household_power_consumption.txt` into the `data/` folder.

### 5. Process the data and train the model

```powershell
python preprocess.py
python train_model.py
```

### 6. Launch the dashboard

```powershell
python -m streamlit run app.py
```

## 📁 Project Structure

```text
PowerWise/
├── app.py
├── preprocess.py
├── train_model.py
├── data/
│   ├── household_power_consumption.txt
│   └── hourly_power.csv
├── powerwise_model.joblib
├── actual_vs_predicted.png
└── README.md
```

Generated files may appear after running the preprocessing and training scripts.

## 🔍 Dashboard Features

- Historical electricity usage visualization
- Actual versus predicted power comparison
- Baseline versus Random Forest evaluation
- Feature importance visualization
- Project workflow and limitations

## ⚠️ Limitations

- The project uses historical household data rather than live smart-meter readings.
- Results depend on the chosen dataset and chronological test split.
- The target is average active power in kilowatts (kW), not energy consumed in kilowatt-hours (kWh).
- Feature importance indicates model reliance, not causation.
- Further validation is needed before using predictions for real-world energy decisions.

## 👩‍💻 Learning Outcomes

This project provided hands-on practice with data cleaning, time-series feature engineering, regression modelling, evaluation metrics, visualization, and building a machine learning dashboard.

## 📌 Future Improvements

- Compare Random Forest with additional regression models.
- Improve time-series validation.
- Add interactive forecast exploration.
- Investigate prediction intervals and model interpretability.
- Deploy the dashboard for public access.

---

**Built as a machine learning learning project using Python and open-source tools.**