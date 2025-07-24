import pandas as pd
import numpy as np
import plotly.express as px
import plotly.colors as pc
import streamlit as st

def calculate_aggregate(df, feature, agg_type, period):
    if period == 'Monthly':
        df['period'] = df['datetime'].dt.to_period('M').astype(str)
    elif period == 'Yearly':
        df['period'] = df['datetime'].dt.to_period('Y').astype(str)
    
    if agg_type == 'Mean':
        agg_df = df.groupby('period')[feature].mean().reset_index()
    elif agg_type == 'Median':
        agg_df = df.groupby('period')[feature].median().reset_index()
    elif agg_type == 'Mode':
        agg_df = df.groupby('period')[feature].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
    elif agg_type == 'Frequency':
        agg_df = df.groupby('period')[feature].size().reset_index(name='frequency')
        feature = 'frequency'
    
    return agg_df, feature

def plot_bar(df, district, feature, agg_type, period, color):
    filtered_data = df[df['name'] == district]
    
    agg_df, plot_feature = calculate_aggregate(filtered_data, feature, agg_type, period)
    
    fig = px.bar(
        agg_df,
        x='period',
        y=plot_feature,
        title=f'{agg_type} {feature.capitalize()} - {period} Aggregation',
        labels={'period': period, plot_feature: feature.capitalize()},
        color=plot_feature,
        color_continuous_scale=color
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        title={
            'text': f'{agg_type} {feature.capitalize()} - {period} Aggregation',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis=dict(
            title=period,
            titlefont=dict(size=18),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=feature.capitalize(),
            titlefont=dict(size=18),
            tickfont=dict(size=14)
        ),
        font=dict(
            family="Arial",
            size=14
        ),
        bargap=0.2,  # Gap between bars
        coloraxis_colorbar=dict(
            title=feature.capitalize(),
            titleside='right'
        )
    )
    
    return fig

def barchart(df):
    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Bar-Polar Chart")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        district = st.selectbox('Select District', df['name'].unique())
    with col2:
        feature = st.selectbox('Select Feature', numcols)
    with col3:
        agg_type = st.selectbox('Select Aggregation Type', ['Mean', 'Median', 'Mode', 'Frequency'])
    with col4:
        period = st.selectbox('Select Time Period', ['Monthly', 'Yearly'])
    with col5:
        colorScale = st.selectbox("Select Color Scale", pc.PLOTLY_SCALES.keys(), index = 11)

    fig = plot_bar(df, district=district, feature=feature, agg_type=agg_type, period=period, color = colorScale)
    st.plotly_chart(fig, use_container_width=True)
