import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import calendar
import plotly.express as px

# Title
st.title("🌦️ Monthly Total Rainfall Analysis")

# Load dataset
@st.cache_data
def load_data():
    ds = xr.open_dataset("Data/RF25_ind2020_rfp25.nc")
    return ds

ds = load_data()

# Processing
daily_sum_rainfall = ds['RAINFALL'].sum(dim=['LATITUDE', 'LONGITUDE'])

sum_rainfall_per_month = daily_sum_rainfall.groupby('TIME.month').sum()

sum_rainfall_df = sum_rainfall_per_month.to_dataframe(name='Total Rainfall (mm)').reset_index()

sum_rainfall_df['Month Name'] = sum_rainfall_df['month'].apply(lambda x: calendar.month_name[x])

sum_rainfall_df = sum_rainfall_df[['Month Name', 'month', 'Total Rainfall (mm)']]

# Display table
st.subheader("📋 Total Rainfall per Month")
st.dataframe(sum_rainfall_df, use_container_width=True)

# Bar plot
st.subheader("📈 Total Monthly Rainfall Chart")

fig = px.bar(
    sum_rainfall_df,
    x='Month Name',
    y='Total Rainfall (mm)',
    text="Total Rainfall (mm)",
    labels={"Total Rainfall (mm)": "Total Rainfall (mm)", "Month Name": "Month"},
    title="Total Rainfall (mm) for Each Month (2020)",
    template='plotly_white'
)

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    xaxis_title="Month",
    yaxis_title="Total Rainfall (mm)",
    title_x=0.5,
    width=900,
    height=500,
    yaxis_tickformat=".2s"
)

st.plotly_chart(fig)
