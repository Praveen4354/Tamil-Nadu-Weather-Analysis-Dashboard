import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def plot_monthly_barpolar(df, district, feature, year, color):
    # Filter data for the selected district and year
    df['datetime'] = pd.to_datetime(df['datetime'])
    filtered_data = df[(df['name'] == district) & (df['datetime'].dt.year == year)]
    
    # Extract month and compute the mean of the feature for each month
    filtered_data['month'] = filtered_data['datetime'].dt.month_name()
    monthly_mean = filtered_data.groupby('month')[feature].mean().reset_index()
    monthly_mean['month'] = pd.Categorical(monthly_mean['month'], 
                                           categories=['January', 'February', 'March', 'April', 'May', 'June', 
                                                       'July', 'August', 'September', 'October', 'November', 'December'], 
                                           ordered=True)
    monthly_mean = monthly_mean.sort_values('month')
    
    # Create the barpolar plot
    plot = go.Figure()
    plot.add_trace(go.Barpolar(
        r=monthly_mean[feature],
        theta=monthly_mean['month'],
        name='Monthly Mean',
        marker=dict(
            color=monthly_mean[feature],
            colorscale=color,
            colorbar=dict(title='Mean Value')
        )
    ))
    
    plot.update_layout(
        title=f"Monthly {feature.capitalize()} for {year}",
        font_size=15,
        polar=dict(
            angularaxis=dict(direction='clockwise')
        ),
        showlegend=False,
        height = 700,
        width = 700
    )
    
    return plot

def barpolar(df):

    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Bar-Polar Chart")

    df['datetime'] = pd.to_datetime(df['datetime'])

    numerical_features = df.select_dtypes(include=[np.number]).columns

    col1, col2, col3 = st.columns(3)

    with col1:
        district = st.selectbox('Select District', df['name'].unique())
    
    with col2:
        years = df['datetime'].dt.year.unique()
        year = st.selectbox('Select Year', years)

    with col3:
        feature = st.selectbox('Select Feature', numerical_features)

    fig = plot_monthly_barpolar(df, district=district, feature=feature, year=year, color = "viridis")
    st.plotly_chart(fig, use_container_width=True)
