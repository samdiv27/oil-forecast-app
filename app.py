import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ======================
# APP CONFIGURATION
# ======================
st.set_page_config(
    page_title="Oil Well Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🛢️ Oil Production Forecast Dashboard")

# ======================
# DATA GENERATION
# ======================
def generate_well_data():
    """Create synthetic well production data"""
    np.random.seed(42)
    wells = ["Well_A", "Well_B", "Well_C"]
    months = np.arange(1, 25)  # 2 years of data
    
    data = []
    for well in wells:
        qi = np.random.uniform(800, 1500)  # Initial production
        di = np.random.uniform(0.05, 0.2)  # Decline rate
        b = np.random.uniform(0.5, 1.5)    # Decline exponent
        
        # Arps' hyperbolic decline formula
        production = qi / (1 + b * di * months) ** (1 / b)
        production += np.random.normal(0, 30, len(months))  # Add noise
        
        for m, p in zip(months, production):
            data.append({
                "Well": well,
                "Month": m,
                "Production": max(0, p),  # No negative values
                "Decline_Rate": di,
                "b_Factor": b
            })
    
    return pd.DataFrame(data)

df = generate_well_data()

# ======================
# SIDEBAR CONTROLS
# ======================
st.sidebar.header("Controls")
selected_well = st.sidebar.selectbox(
    "Select Well", 
    df["Well"].unique()
)
forecast_period = st.sidebar.slider(
    "Forecast Months", 
    1, 12, 6
)

# ======================
# MAIN DISPLAY
# ======================
tab1, tab2, tab3 = st.tabs(["📈 Production", "🔮 Forecast", "💰 Economics"])

# TAB 1: Production History
with tab1:
    st.header(f"Production History: {selected_well}")
    well_data = df[df["Well"] == selected_well]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(well_data["Month"], well_data["Production"], "b-o")
    ax.set_xlabel("Month")
    ax.set_ylabel("Production (bbl/month)")
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Well Parameters")
    st.write(f"""
    - **Initial Production (qi):** {well_data["Production"].iloc[0]:.1f} bbl/month
    - **Decline Rate (Di):** {well_data["Decline_Rate"].iloc[0]:.3f}/month
    - **b-Factor:** {well_data["b_Factor"].iloc[0]:.3f}
    """)

# TAB 2: Forecast
with tab2:
    st.header("Machine Learning Forecast")
    
    # Feature Engineering
    df["Cumulative"] = df.groupby("Well")["Production"].cumsum()
    df["Moving_Avg"] = df.groupby("Well")["Production"].rolling(3).mean().reset_index(level=0, drop=True)
    
    # Prepare data
    X = df[["Month", "Decline_Rate", "b_Factor", "Cumulative", "Moving_Avg"]]
    y = df["Production"]
    
    # Create pipeline with imputer and model
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handles missing values
        ('regressor', RandomForestRegressor(n_estimators=100))
    ])
    
    # Train model
    model.fit(X, y)
    
    # Generate Forecast
    last_data = df[df["Well"] == selected_well].iloc[-1]
    future = pd.DataFrame({
        "Month": range(last_data["Month"] + 1, last_data["Month"] + 1 + forecast_period),
        "Decline_Rate": last_data["Decline_Rate"],
        "b_Factor": last_data["b_Factor"],
        "Cumulative": [last_data["Cumulative"] + last_data["Production"] * i for i in range(1, forecast_period + 1)],
        "Moving_Avg": last_data["Moving_Avg"]
    }).fillna(0)  # Ensure no NaN values
    
    future["Forecast"] = model.predict(future[X.columns])
    
    # Plot Forecast
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(well_data["Month"], well_data["Production"], "b-o", label="Historical")
    ax2.plot(future["Month"], future["Forecast"], "r--o", label="Forecast")
    ax2.set_title(f"{forecast_period}-Month Production Forecast")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# TAB 3: Economics
with tab3:
    st.header("Economic Analysis")
    oil_price = st.number_input("Oil Price ($/bbl)", value=75.0)
    operating_cost = st.number_input("Operating Cost ($/bbl)", value=20.0)
    
    future["Revenue"] = future["Forecast"] * oil_price
    future["Profit"] = future["Forecast"] * (oil_price - operating_cost)
    
    st.subheader("Projected Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Production", f"{future['Forecast'].sum():.1f} bbl")
    with col2:
        st.metric("Gross Revenue", f"${future['Revenue'].sum():,.0f}")
    with col3:
        st.metric("Net Profit", f"${future['Profit'].sum():,.0f}")

# ======================
# FOOTER
# ======================
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit | For educational purposes")