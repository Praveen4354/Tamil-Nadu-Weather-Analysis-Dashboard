import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st

def map_season(month):
    if month in [3, 4]:
        return 1
    elif month == 5:
        return 2
    elif month in [6, 7]:
        return 3
    elif month in [7, 8]:
        return 4
    elif month in [9, 10]:
        return 5
    elif month in [11, 12]:
        return 6
    elif month in [1, 2]:
        return 7

def plot_seasons(master, district, frequency, feature):
    master['season'] = master['datetime'].dt.month.apply(map_season)
    
    data = master[master['name'] == district]
    
    ts_data = data.set_index('datetime')[feature].asfreq('D').interpolate()
    decomp = seasonal_decompose(ts_data, period=frequency)
    
    trend = decomp.trend
    dates = data['datetime']
    
    fig = go.Figure()

    seasonNames = ['Summer', 'Summer (Pre-Monsoon)', 'Southwest Monsoon', 'Southwest Monsoon (Retreating)', 'Northeast Monsoon', 'Winter', 'Summer (Pre-Monsoon)']
    
    for season in data['season'].unique():
        season_data = data[data['season'] == season]
        fig.add_trace(go.Scatter(
            x=season_data['datetime'], 
            y=season_data[feature], 
            mode='markers', 
            name=seasonNames[season - 1]
        ))
    
    fig.add_trace(go.Scatter(
        x=dates, 
        y=trend, 
        mode='lines', 
        name='Trend', 
        line=dict(color='white', width=3)
    ))
    
    fig.update_layout(
        title=f'{feature} Variation by Season in {district}',
        height = 550,
        xaxis_title='Date',
        yaxis_title=feature,
        legend_title='Season'
    )
    
    return fig

def seasonalScatter(df):
    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Heatmap Plot")
    with st.expander("Show code"):
        st.code("Code")
    
    districts = df["name"].unique()
    col1, col2, col3 = st.columns(3)

    with col1:   
        featureX = st.selectbox("Select Feature", numcols, key = "sf1")
    with col2:
        dist = st.selectbox("Select District:",districts, key = 'sf2') 
    with col3:
        freq = st.number_input("Insert a number",min_value = 10, max_value = 100, step=10)

    if featureX:
        st.plotly_chart(plot_seasons(df, district=dist, frequency=freq, feature=featureX), use_container_width=True)