import streamlit as st
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import geopandas as gpd
from shapely.geometry import box,point
import numpy as np
from geopy.geocoders import Nominatim
import plotly.graph_objects as go
from pandas.api.types import CategoricalDtype
import calendar 

@st.cache_data
def load_dataset():
    return xr.open_dataset("Data/RF25_ind2020_rfp25.nc")

def lat_long_extract(address):
    geolocator = Nominatim(user_agent="address_geocoder")
    location = geolocator.geocode(address)
    
    if location is None:
        return None, None, None, None

    latitude_exact = location.latitude
    longitude_exact = location.longitude
    
    def round_to_quarter(x):
        return round(x * 4) / 4

    latitude_rounded = round_to_quarter(latitude_exact)
    longitude_rounded = round_to_quarter(longitude_exact)

    return latitude_exact, longitude_exact, latitude_rounded, longitude_rounded

def get_bound_box(dist_lat, dist_long):
    lat_range = [dist_lat - 0.25, dist_lat + 0.25]
    lon_range = [dist_long - 0.25, dist_long + 0.25]
    return lat_range, lon_range

st.title("📍 Get Coordinates from Address")

address = st.text_input("Enter the location or Pincode")

if address:
    lat_exact, long_exact, lat_rounded, long_rounded = lat_long_extract(address)
    
    if lat_exact is None:
        st.error("❌ Could not find the location. Please check the address.")
    else:
        st.success("✅ Exact Latitude and Longitude")
        st.code(f"Latitude: {lat_exact}\nLongitude: {long_exact}", language='python')

        st.info("✅ Rounded Latitude and Longitude (to nearest 0.25)")
        st.code(f"Latitude: {lat_rounded}\nLongitude: {long_rounded}", language='python')

        st.header(f"Average Rainfall for {address} - Method 1 (Single Point)")

        ds = load_dataset()

        input_date = '2020-01-01'
        rain_slice = ds['RAINFALL'].sel(TIME=input_date)

        lats = ds['LATITUDE'].values
        lons = ds['LONGITUDE'].values
        rain = rain_slice.values

        fig = go.Figure(data=go.Heatmap(
            z=rain,
            x=lons,
            y=lats,
            colorscale='Blues',
            showscale=False
        ))

        fig.add_trace(go.Scattergl(
            x=[long_rounded],  
            y=[lat_rounded],   
            mode='markers',
            marker=dict(
                color='red',
                size=8,
                symbol='cross'
            ),
            name='Location'
        ))

        fig.update_layout(
            title=dict(
                text=f'Location of {address} (Rounded)',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            width=800,
            height=700,
        )
        
        st.plotly_chart(fig)

        # --- Daily Rainfall Extraction ---
        dist_data = ds.sel(LATITUDE=lat_rounded, LONGITUDE=long_rounded)
        dist_daily_rainfall = dist_data['RAINFALL']

        df = pd.DataFrame({
            "Date": pd.to_datetime(ds['TIME'].values),
            "Rainfall_mm": dist_daily_rainfall.values
        })

        dist_daily_avg_rainfall = df['Rainfall_mm'].mean()

        st.success(f"📈 The average rainfall for **{address}** in 2020 is **{dist_daily_avg_rainfall:.2f} mm**")

        # --- Line Plot for Daily Rainfall ---
        fig2 = px.line(
            df,
            x='Date',
            y='Rainfall_mm',
            title=f'Daily Rainfall for {address} (2020)',
            labels={'Rainfall_mm': 'Rainfall in mm', 'Date': 'Date'},
            template='plotly_white'
        )

        fig2.update_layout(
            width=1000,
            height=400,
            title_x=0.5,
            xaxis_title='Date',
            yaxis_title='Rainfall in mm',
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )

        st.plotly_chart(fig2)

        st.header(f"Average Rainfall for {address} - Method 2 (Single Point)")

        ds = load_dataset()

        
        dist_lat_rng, dist_long_rng = get_bound_box(lat_rounded, long_rounded)

        
        rain_slice = ds['RAINFALL'].sel(TIME='2020-01-01')  # First day example
        lats = ds['LATITUDE'].values
        lons = ds['LONGITUDE'].values
        rain = rain_slice.values

        fig = go.Figure(data=go.Heatmap(
            z=rain,
            x=lons,
            y=lats,
            colorscale='Blues',
            showscale=False
        ))

        fig.add_shape(
            type="rect",
            x0=dist_long_rng[0], y0=dist_lat_rng[0],
            x1=dist_long_rng[1], y1=dist_lat_rng[1],
            line=dict(
                color="red",
                width=3,
            ),
        )
        fig.add_trace(go.Scattergl(
            x=[long_rounded],  
            y=[lat_rounded],   
            mode='markers',
            marker=dict(
                color='blue',
                size=8,
                symbol='cross'  
            ),
            name='Location'
        ))

        fig.update_layout(
            title=dict(
                text=f'Location of {address} - Method-1 vs Method-2 (Bounding Box)',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            width=800,
            height=700,
        )

        st.plotly_chart(fig)

        
        dist_data_2 = ds.sel(
            LATITUDE=slice(dist_lat_rng[0], dist_lat_rng[1]),
            LONGITUDE=slice(dist_long_rng[0], dist_long_rng[1])
        )

        avg_rain_each_day = dist_data_2['RAINFALL'].mean(dim=['LATITUDE', 'LONGITUDE'])

        df_avg_rainfall = pd.DataFrame({
            'Date': dist_data_2['TIME'].values,
            'Avg_Rainfall_mm': avg_rain_each_day.values
        })

        avg_rainfall = df_avg_rainfall['Avg_Rainfall_mm'].mean()

        st.success(f"📊 The average rainfall for **{address}** (bounding box) in 2020 is **{avg_rainfall:.2f} mm**")

        
        fig2 = px.line(
            df_avg_rainfall,
            x='Date',
            y='Avg_Rainfall_mm',
            title=f'Daily Average Rainfall for {address} (Bounding Box) (2020)',
            labels={'Avg_Rainfall_mm': 'Rainfall in mm', 'Date': 'Date'},
            template='plotly_white'
        )

        fig2.update_layout(
            width=1000,
            height=400,
            title_x=0.5,
            xaxis_title='Date',
            yaxis_title='Rainfall in mm',
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )

        st.plotly_chart(fig2)

        st.subheader("📈 Monthly Rainfall Statistics for District")

        df['Month'] = pd.to_datetime(df['Date']).dt.month_name()

        month_order = pd.CategoricalDtype(
            ['January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'],
            ordered=True
        )

        monthly_avg_dist = df.groupby('Month')['Rainfall_mm'].mean().reset_index()
        monthly_avg_dist['Month'] = monthly_avg_dist['Month'].astype(month_order)
        monthly_avg_dist = monthly_avg_dist.sort_values('Month')
        monthly_avg_dist.reset_index(drop=True)

        monthly_tot_dist = df.groupby('Month')['Rainfall_mm'].sum().reset_index()
        monthly_tot_dist['Month'] = monthly_tot_dist['Month'].astype(month_order)
        monthly_tot_dist = monthly_tot_dist.sort_values('Month')
        monthly_tot_dist.reset_index(drop=True)

        monthly_max_dist = df.groupby('Month')['Rainfall_mm'].max().reset_index()
        monthly_max_dist['Month'] = monthly_max_dist['Month'].astype(month_order)
        monthly_max_dist = monthly_max_dist.sort_values('Month')
        monthly_max_dist.reset_index(drop=True)

        tab1, tab2, tab3 = st.tabs(["📊 Monthly Average", "📊 Monthly Total", "📊 Monthly Maximum"])

        with tab1:
            st.dataframe(monthly_avg_dist.style.hide(axis='index'), use_container_width=True)
            csv = monthly_avg_dist.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇️ Download Average Rainfall {address} CSV",
                data=csv,
                file_name=f'monthly_avg_rainfall_{address}.csv',
                mime='text/csv',
            )

        with tab2:
            st.dataframe(monthly_tot_dist.style.hide(axis='index'), use_container_width=True)
            csv = monthly_tot_dist.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇️ Download Total Rainfall {address} CSV",
                data=csv,
                file_name=f'monthly_total_rainfall_{address}.csv',
                mime='text/csv',
            )

        with tab3:
            st.dataframe(monthly_max_dist.style.hide(axis='index'), use_container_width=True)
            csv = monthly_max_dist.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇️ Download Maximum Rainfall {address} CSV",
                data=csv,
                file_name=f'monthly_max_rainfall_{address}.csv',
                mime='text/csv',
            )

