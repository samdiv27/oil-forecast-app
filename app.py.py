import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# App title
st.title("Oil Well Production Forecast")

# Sample data generation
def generate_sample_well():
    months = list(range(1, 25))
    production = [1000 / (1 + 0.1 * month) for month in months]  # Simple decline
    return pd.DataFrame({"Month": months, "Production": production})

df = generate_sample_well()

# Plot
st.subheader("Production Decline Curve")
fig, ax = plt.subplots()
ax.plot(df["Month"], df["Production"], 'b-')
ax.set_xlabel("Month")
ax.set_ylabel("Production (bbl)")
st.pyplot(fig)

# Basic forecast
st.subheader("6-Month Forecast")
st.write("Next 6 months prediction:")
forecast = pd.DataFrame({
    "Month": [25, 26, 27, 28, 29, 30],
    "Predicted": [50, 45, 40, 36, 32, 29]  # Example values
})
st.bar_chart(forecast.set_index("Month"))