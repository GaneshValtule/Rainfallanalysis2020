import streamlit as st
import xarray as xr
import numpy as np
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from datetime import datetime

st.set_page_config(
    page_title="Rainfall Analysis 2020",
    page_icon="🌧️",
)

st.title("Welcome to the Rainfall Analysis Dashboard for India")

@st.cache_data
def load_dataset():
    return xr.open_dataset("Data/RF25_ind2020_rfp25.nc")

ds = load_dataset()
available_dates = ds['TIME'].dt.strftime('%Y-%m-%d').values

st.title("🌧️ Rainfall Heatmap Viewer")



month = st.selectbox("Select a month", range(1, 13), format_func=lambda x: datetime(2020, x, 1).strftime('%B'))
day = st.slider("Select a day", min_value=1, max_value=31, value=1)
year = st.selectbox("Select a year", [2020]) 


selected_date = f"{year}-{month:02d}-{day:02d}"


st.write(f"Selected Date: {selected_date}")


rain_slice = ds['RAINFALL'].sel(TIME=selected_date)
lats = ds['LATITUDE'].values
lons = ds['LONGITUDE'].values
rain = rain_slice.values


fig = go.Figure(data=go.Heatmap(
    z=rain,
    x=lons,
    y=lats,
    colorscale='Blues',
    colorbar=dict(title='Rainfall (mm)'),
    zmin=np.nanmin(rain),
    zmax=np.nanmax(rain)
))

fig.update_layout(
    title=dict(
        text=f'Rainfall on {selected_date}',
        x=0.5,
        xanchor='center'
    ),
    xaxis_title='Longitude',
    yaxis_title='Latitude',
    xaxis=dict(scaleanchor='y')
    width=1000,
    height=700,
)

st.plotly_chart(fig, use_container_width=True)
