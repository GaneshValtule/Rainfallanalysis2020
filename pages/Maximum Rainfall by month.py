import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import calendar
import plotly.express as px

# Title
st.title("🌧️ Monthly Maximum Rainfall Analysis")

# Load dataset
@st.cache_data
def load_data():
    ds = xr.open_dataset("Data/RF25_ind2020_rfp25.nc")
    return ds

ds = load_data()

# Processing
daily_max_rainfall = ds['RAINFALL'].max(dim=['LATITUDE', 'LONGITUDE'])

max_rainfall_per_month = daily_max_rainfall.groupby('TIME.month').max()

months = range(1, 13)
max_rainfall_data = []

for month in months:
    month_data = ds['RAINFALL'].sel(TIME=ds['TIME'].dt.month == month)
    day_of_max = month_data.max(dim=['LATITUDE', 'LONGITUDE']).argmax(dim='TIME').item()
    
    rainfall_on_max_day = month_data.isel(TIME=day_of_max)
    idx = np.unravel_index(np.nanargmax(rainfall_on_max_day.values), rainfall_on_max_day.shape)
    lat_of_max = rainfall_on_max_day['LATITUDE'].values[idx[0]]
    lon_of_max = rainfall_on_max_day['LONGITUDE'].values[idx[1]]
    max_rainfall = rainfall_on_max_day.values[idx]
    
    max_rainfall_data.append((month, day_of_max, max_rainfall, lat_of_max, lon_of_max))

# DataFrame
df = pd.DataFrame(max_rainfall_data, columns=['Month', 'Date', 'Max Rainfall (mm)', 'Latitude', 'Longitude'])
df['Month Name'] = df['Month'].apply(lambda x: calendar.month_name[x])

# Reorder columns
df = df[['Month Name', 'Month', 'Date', 'Max Rainfall (mm)', 'Latitude', 'Longitude']]

# Show Table
st.subheader("📋 Maximum Rainfall Details")
st.dataframe(df, use_container_width=True)

# Plotly Visualization
st.subheader("📈 Maximum Rainfall per Month")

fig = px.bar(
    df,
    x="Month Name",
    y="Max Rainfall (mm)",
    hover_data=["Month Name", "Date", "Latitude", "Longitude"],
    labels={"Max Rainfall (mm)": "Max Rainfall (mm)", "Month Name": "Month"},
    title="Monthly Maximum Rainfall with latitude and Longitude"
)

st.plotly_chart(fig)
