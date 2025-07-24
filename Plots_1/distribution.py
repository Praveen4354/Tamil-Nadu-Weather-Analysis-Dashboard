import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, kurtosis, norm
from collections import Counter
import streamlit as st
import plotly.figure_factory as ff

def univariate_analysis(data, featureX, bin, district, color):

    data = data[data['name'] == district]

    mean = np.mean(data[featureX])
    median = np.median(data[featureX])
    mode = Counter(data[featureX]).most_common(1)[0][0]
    skewness = skew(data[featureX])
    kurt = kurtosis(data[featureX])

    fig = ff.create_distplot([data[featureX]], [featureX], curve_type='normal', colors=['#636EFA'])

    fig.add_vline(x=mean, line=dict(color='white', dash = 'dash', width = 3), name='Mean')
    fig.add_vline(x=median, line=dict(color='yellow', dash = 'dash', width = 3), name='Median')
    fig.add_vline(x=mode, line=dict(color='orange', dash = 'dash', width = 3), name='Mode')


    fig.update_layout(xaxis_title=featureX, yaxis_title='Density', template='plotly_white', height = 700, showlegend = True)

    table_data = {
        "Statistic": ["Mean", "Median", "Mode", "Skewness", "Kurtosis"],
        "Value": [mean, median, mode, skewness, kurt]
    }

    st.table(table_data)

    return fig

def distribution(df):
    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Heatmap Plot")
    with st.expander("Show code"):
        st.code("Code")
    
    districts = df["name"].unique()
    col1, col2, col3, col4 = st.columns(4)

    with col1:   
        featureX = st.selectbox("Select Feature", numcols, key = "sf1")
    with col2:
        dist = st.selectbox("Select District:", districts, key = 'sf2') 
    with col3:
        freq = st.number_input("Number of bins: ", min_value = 10, max_value = 100, step=10)
    with col4:
        color = st.color_picker("Pick Color: ", value="#06ACFF")

    if featureX:
        st.plotly_chart(univariate_analysis(df, featureX, freq, dist, color), use_container_width=True)