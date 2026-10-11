
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="MotorGuard | Health Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0b1220;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111c30;
        border-right: 1px solid #263449;
    }

    h1, h2, h3 {
        color: #f1f5f9 !important;
    }

    p, label, span {
        color: inherit;
    }

    [data-testid="stMetric"] {
        background-color: #142238;
        border: 1px solid #263b54;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #38bdf8;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    .hero {
        background: linear-gradient(120deg, #142b46, #123b49);
        padding: 26px;
        border-radius: 16px;
        border: 1px solid #28536a;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 30px;
        font-weight: 700;
        color: #f8fafc;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 15px;
    }

    .small-note {
        color: #94a3b8;
        font-size: 13px;
    }

    div.stButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DATA GENERATION
# --------------------------------------------------
@st.cache_data
def create_data():
    rng = np.random.default_rng(42)
    n = 600

    hours = rng.uniform(0, 12, n)
    load = rng.uniform(20, 100, n)
    ambient = rng.uniform(18, 38, n)

    temperature = (
        25
        + 0.75 * hours
        + 0.16 * load
        + 0.45 * ambient
        + rng.normal(0, 2.5, n)
    )

    return pd.DataFrame({
        "Operating Hours": hours,
        "Motor Load (%)": load,
        "Ambient Temperature (°C)": ambient,
        "Motor Temperature (°C)": temperature
    })


data = create_data()

features = [
    "Operating Hours",
    "Motor Load (%)",
    "Ambient Temperature (°C)"
]

X = data[features]
y = data["Motor Temperature (°C)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙️ MotorGuard")
st.sidebar.caption("Predictive Maintenance Dashboard")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Temperature Analysis",
        "Model Performance",
        "Dataset Explorer"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Motor Conditions")

hours = st.sidebar.slider(
    "Operating Hours",
    0.0, 12.0, 5.0, 0.5
)

load = st.sidebar.slider(
    "Motor Load (%)",
    20, 100, 60, 5
)

ambient = st.sidebar.slider(
    "Ambient Temperature (°C)",
    18, 38, 28, 1
)

st.sidebar.divider()
st.sidebar.caption("Model: Multiple Linear Regression")
st.sidebar.caption("Data: Synthetic demonstration dataset")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
motor_input = pd.DataFrame({
    "Operating Hours": [hours],
    "Motor Load (%)": [load],
    "Ambient Temperature (°C)": [ambient]
})

predicted_temp = float(model.predict(motor_input)[0])

# Illustrative categories only, not validated safety limits
if predicted_temp < 60:
    status = "Normal range (illustrative)"
elif predicted_temp < 80:
    status = "Elevated temperature (illustrative)"
else:
    status = "High temperature (illustrative)"

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">⚙️ Motor Health Trend Predictor</div>
    <div class="hero-subtitle">
        Machine learning dashboard for electric motor
        temperature estimation and predictive maintenance research.
    </div>
</div>
""", unsafe_allow_html=True)

st.warning(
    "DEMO MODE: All sensor data is synthetic. "
    "Predictions and temperature categories are illustrative, "
    "not real-world safety assessments."
)

# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------
if page == "Overview":

    st.subheader("Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Estimated Temperature",
        f"{predicted_temp:.2f} °C"
    )

    c2.metric(
        "Motor Load",
        f"{load}%"
    )

    c3.metric(
        "Mean Absolute Error",
        f"{mae:.2f} °C"
    )

    c4.metric(
        "R² Score",
        f"{r2:.3f}"
    )

    st.subheader("Current Motor Conditions")

    a, b = st.columns(2)

    with a:
        st.write("**Operating Hours**")
        st.progress(min(hours / 12, 1.0))
        st.write(f"{hours:.1f} of 12 hours")

        st.write("**Motor Load**")
        st.progress(load / 100)
        st.write(f"{load}% load")

    with b:
        st.write("**Ambient Temperature**")
        st.progress((ambient - 18) / 20)
        st.write(f"{ambient} °C")

        st.write("**Estimated Status**")
        st.info(status)

    st.subheader("Temperature Prediction")

    st.metric(
        "Predicted Motor Temperature",
        f"{predicted_temp:.2f} °C"
    )

    st.write(
        "Change the sliders in the sidebar to explore how "
        "different operating conditions affect the model output."
    )

    st.subheader("Recent Synthetic Sample Data")

    st.dataframe(
        data.head(10).round(2),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# TEMPERATURE ANALYSIS PAGE
# --------------------------------------------------
elif page == "Temperature Analysis":

    st.subheader("📈 Temperature Analysis")

    st.write(
        "Explore the relationship between operating conditions "
        "and the model's estimated motor temperature."
    )

    trend_hours = np.linspace(0, 12, 50)

    trend_data = pd.DataFrame({
        "Operating Hours": trend_hours,
        "Motor Load (%)": [load] * 50,
        "Ambient Temperature (°C)": [ambient] * 50
    })

    trend_temps = model.predict(trend_data)

    trend_df = pd.DataFrame({
        "Operating Hours": trend_hours,
        "Estimated Temperature (°C)": trend_temps
    })

    st.line_chart(
        trend_df.set_index("Operating Hours"),
        use_container_width=True
    )

    st.caption(
        "Simulated trend with motor load and ambient temperature "
        "held constant. This is a model-generated relationship, "
        "not a real sensor time series."
    )

    st.subheader("Temperature vs Motor Load")

    load_values = np.arange(20, 101, 5)

    load_data = pd.DataFrame({
        "Operating Hours": [hours] * len(load_values),
        "Motor Load (%)": load_values,
        "Ambient Temperature (°C)": [ambient] * len(load_values)
    })

    load_temps = model.predict(load_data)

    load_df = pd.DataFrame({
        "Motor Load (%)": load_values,
        "Estimated Temperature (°C)": load_temps
    })

    st.line_chart(
        load_df.set_index("Motor Load (%)"),
        use_container_width=True
    )

    st.subheader("Temperature Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#142238")
    ax.set_facecolor("#142238")

    ax.hist(
        data["Motor Temperature (°C)"],
        bins=25,
        edgecolor="white",
        alpha=0.85
    )

    ax.set_xlabel("Temperature (°C)", color="white")
    ax.set_ylabel("Frequency", color="white")
    ax.set_title("Synthetic Motor Temperature Distribution", color="white")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#64748b")

    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------
elif page == "Model Performance":

    st.subheader("🤖 Model Performance")

    m1, m2, m3 = st.columns(3)

    m1.metric("MAE", f"{mae:.3f} °C")
    m2.metric("RMSE", f"{rmse:.3f} °C")
    m3.metric("R² Score", f"{r2:.3f}")

    st.write(
        "**MAE:** Average absolute prediction error. "
        "Lower is better."
    )

    st.write(
        "**RMSE:** Error measure that penalizes larger errors more. "
        "Lower is better."
    )

    st.write(
        "**R²:** Measures how well the model explains variation "
        "in the test target. Interpret it in the context of this "
        "synthetic dataset."
    )

    st.subheader("Actual vs Predicted")

    comparison = pd.DataFrame({
        "Actual Temperature (°C)": y_test.to_numpy(),
        "Predicted Temperature (°C)": predictions
    })

    st.scatter_chart(
        comparison,
        x="Actual Temperature (°C)",
        y="Predicted Temperature (°C)",
        use_container_width=True
    )

    st.subheader("Prediction Error")

    errors = y_test.to_numpy() - predictions

    error_df = pd.DataFrame({
        "Prediction Error (°C)": errors
    })

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#142238")
    ax.set_facecolor("#142238")

    ax.hist(errors, bins=25, edgecolor="white", alpha=0.85)
    ax.axvline(0, linestyle="--", linewidth=2)

    ax.set_xlabel("Actual - Predicted (°C)", color="white")
    ax.set_ylabel("Frequency", color="white")
    ax.set_title("Distribution of Prediction Errors", color="white")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#64748b")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature Coefficients")

    coefficient_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    })

    st.dataframe(
        coefficient_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.write(f"Model intercept: {model.intercept_:.3f}")

    st.caption(
        "Coefficients describe the fitted linear relationship. "
        "Their magnitudes should not be compared directly without "
        "considering the different feature units."
    )

# --------------------------------------------------
# DATASET EXPLORER PAGE
# --------------------------------------------------
elif page == "Dataset Explorer":

    st.subheader("🗂️ Dataset Explorer")

    st.write(
        "Inspect the generated dataset, filter rows, "
        "and export the data for further analysis."
    )

    st.metric("Total Records", len(data))
    st.metric("Input Features", len(features))

    min_load, max_load = st.slider(
        "Filter by Motor Load (%)",
        min_value=20,
        max_value=100,
        value=(20, 100)
    )

    filtered_data = data[
        (data["Motor Load (%)"] >= min_load)
        & (data["Motor Load (%)"] <= max_load)
    ]

    st.write(f"Matching records: {len(filtered_data)}")

    st.dataframe(
        filtered_data.round(2),
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="motor_health_dataset.csv",
        mime="text/csv"
    )

    report = pd.DataFrame({
        "Metric": [
            "Total Records",
            "Average Motor Temperature (°C)",
            "Maximum Motor Temperature (°C)",
            "Minimum Motor Temperature (°C)",
            "MAE (°C)",
            "RMSE (°C)",
            "R² Score"
        ],
        "Value": [
            len(data),
            data["Motor Temperature (°C)"].mean(),
            data["Motor Temperature (°C)"].max(),
            data["Motor Temperature (°C)"].min(),
            mae,
            rmse,
            r2
        ]
    })

    st.download_button(
        label="Download Model Summary (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="motor_health_model_summary.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "MotorGuard | Motor Health Trend Predictor | "
    "Python • Streamlit • Pandas • Scikit-learn"
)

st.caption(
    "Academic prototype using synthetic data. "
    "Not validated for operational or safety-critical decisions."
)
