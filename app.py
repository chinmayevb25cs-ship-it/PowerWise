
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "hourly_power.csv"
MODEL_PATH = BASE_DIR / "powerwise_model.joblib"

st.set_page_config(
    page_title="PowerWise | Energy Forecasting",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #16a34a;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 24px;
    }
    .section-note {
        color: #64748b;
        font-size: 14px;
    }
    div[data-testid="stMetric"] {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 16px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚡ PowerWise</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Household Electricity Consumption Forecasting</p>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    return data.set_index("datetime").sort_index()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    df = load_data()
    saved = load_model()
except Exception as error:
    st.error(f"Could not load the dataset or model: {error}")
    st.stop()

model = saved["model"]
features = saved["features"]

# --------------------------------------------------
# RECREATE FEATURES FOR EVALUATION
# --------------------------------------------------

data = df.copy()

data["hour"] = data.index.hour
data["day_of_week"] = data.index.day_of_week
data["month"] = data.index.month

for lag in [1, 2, 3, 24, 48, 168]:
    data[f"lag_{lag}"] = data["Global_active_power"].shift(lag)

data["target"] = data["Global_active_power"].shift(-24)
data = data.dropna()

X = data[features]
actual = data["target"]

split_idx = int(len(data) * 0.8)

X_test = X.iloc[split_idx:]
y_test = actual.iloc[split_idx:]

predictions = model.predict(X_test)
baseline = X_test["lag_24"].to_numpy()

model_mae = mean_absolute_error(y_test, predictions)
model_rmse = np.sqrt(mean_squared_error(y_test, predictions))

baseline_mae = mean_absolute_error(y_test, baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline))

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("⚡ PowerWise")
st.sidebar.caption("ML Energy Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "Forecasting Dashboard",
        "Model Comparison",
        "About the Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Built with Python, Pandas, Scikit-learn and Streamlit")

# --------------------------------------------------
# PAGE 1: FORECASTING DASHBOARD
# --------------------------------------------------

if page == "Forecasting Dashboard":

    st.header("📈 Forecasting Dashboard")
    st.write(
        "Explore historical household electricity usage "
        "and inspect the model's predictions."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Hourly readings", f"{len(df):,}")
    c2.metric(
        "Average power",
        f"{df['Global_active_power'].mean():.2f} kW"
    )
    c3.metric("Random Forest MAE", f"{model_mae:.4f} kW")

    st.divider()

    st.subheader("Historical Electricity Usage")

    history = df.iloc[-24 * 14:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history.index,
        y=history["Global_active_power"],
        mode="lines",
        name="Average active power",
        line=dict(width=2)
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average active power (kW)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Actual vs Predicted Power")

    sample_actual = y_test.iloc[:168]
    sample_predictions = predictions[:168]

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=sample_actual.index,
        y=sample_actual,
        mode="lines",
        name="Actual"
    ))

    fig2.add_trace(go.Scatter(
        x=sample_actual.index,
        y=sample_predictions,
        mode="lines",
        name="Random Forest prediction"
    ))

    fig2.update_layout(
        xaxis_title="Date",
        yaxis_title="Average active power (kW)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "The prediction chart shows a sample from the chronological test set. "
        "These are 24-hour-ahead targets evaluated at hourly timestamps."
    )

# --------------------------------------------------
# PAGE 2: MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.header("📊 Model Comparison")
    st.write(
        "Compare a simple previous-day baseline with the Random Forest model."
    )
    st.caption("Lower MAE and RMSE values indicate smaller prediction errors.")

    # Metrics table
    comparison = pd.DataFrame({
        "Model": [
            "Previous-day baseline",
            "Random Forest"
        ],
        "MAE (kW)": [
            baseline_mae,
            model_mae
        ],
        "RMSE (kW)": [
            baseline_rmse,
            model_rmse
        ]
    })

    st.dataframe(
        comparison.style.format({
            "MAE (kW)": "{:.4f}",
            "RMSE (kW)": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Metric cards
    st.subheader("Performance Summary")

    col1, col2 = st.columns(2)

    mae_improvement = (
        (baseline_mae - model_mae) / baseline_mae * 100
        if baseline_mae != 0 else 0
    )

    rmse_improvement = (
        (baseline_rmse - model_rmse) / baseline_rmse * 100
        if baseline_rmse != 0 else 0
    )

    col1.metric(
        "Random Forest MAE",
        f"{model_mae:.4f} kW",
        delta=f"{mae_improvement:.1f}% lower than baseline"
    )

    col2.metric(
        "Random Forest RMSE",
        f"{model_rmse:.4f} kW",
        delta=f"{rmse_improvement:.1f}% lower than baseline"
    )

    # Error comparison chart
    st.subheader("Error Comparison")

    chart = go.Figure()

    chart.add_trace(go.Bar(
        x=["MAE", "RMSE"],
        y=[baseline_mae, baseline_rmse],
        name="Previous-day baseline"
    ))

    chart.add_trace(go.Bar(
        x=["MAE", "RMSE"],
        y=[model_mae, model_rmse],
        name="Random Forest"
    ))

    chart.update_layout(
        barmode="group",
        xaxis_title="Evaluation metric",
        yaxis_title="Error (kW)",
        template="plotly_white",
        legend_title="Model"
    )

    st.plotly_chart(chart, use_container_width=True)

    st.success(
        f"On this test split, Random Forest reduced MAE by "
        f"{mae_improvement:.1f}% compared with the baseline."
    )

    # Feature importance
    st.divider()
    st.subheader("🔍 Feature Importance")

    st.write(
        "Feature importance shows how much each input feature contributes "
        "to the Random Forest's learned predictions. It does not prove "
        "that a feature causes electricity usage to change."
    )

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    importance_chart = go.Figure()

    importance_chart.add_trace(go.Bar(
        x=importance["Importance"],
        y=importance["Feature"],
        orientation="h",
        name="Feature importance"
    ))

    importance_chart.update_layout(
        xaxis_title="Relative importance",
        yaxis_title="Input feature",
        template="plotly_white"
    )

    st.plotly_chart(importance_chart, use_container_width=True)

    st.caption(
        "Importance values are model-specific and should not be interpreted "
        "as causal effects."
    )

# --------------------------------------------------
# PAGE 3: ABOUT THE PROJECT
# --------------------------------------------------

elif page == "About the Project":

    st.header("💡 About PowerWise")

    st.write(
        "PowerWise is a machine learning project that explores historical "
        "household electricity consumption and evaluates power forecasts."
    )

    st.subheader("Dataset")

    st.markdown("""
    - **Source:** UCI Individual Household Electric Power Consumption
    - **Original data:** Minute-level household measurements
    - **Processed data:** Hourly average active power
    - **Target:** Average active power 24 hours ahead
    """)

    st.subheader("Machine Learning Workflow")

    st.markdown("""
    1. Load and inspect the raw dataset.
    2. Handle missing power readings.
    3. Aggregate minute-level readings into hourly averages.
    4. Create time-based and lag features.
    5. Train a Random Forest regression model.
    6. Compare against a previous-day baseline.
    7. Evaluate using MAE and RMSE.
    """)

    st.subheader("Tools Used")

    st.markdown("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Plotly
    - Streamlit
    """)

    st.subheader("Limitations")

    st.markdown("""
    - The model uses historical household data, not a live electricity meter.
    - Results depend on the chronological test split.
    - The forecast target is average power in kW, not energy consumption in kWh.
    - Feature importance indicates model reliance, not causation.
    """)
