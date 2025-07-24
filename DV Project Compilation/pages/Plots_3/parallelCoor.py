import streamlit as st
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.express as px

def parallelCoord(df, features1, features2):

    sample_size_per_spi = 10
    sampled_data = []

    for spi_value in range(1,16):
        spi_samples = df[df['SPI'] == spi_value].sample(sample_size_per_spi, replace=True)
        sampled_data.append(spi_samples)

    sample1 = pd.concat(sampled_data, ignore_index=True)
    sample1 = sample1.sample(frac=1).reset_index(drop=True)
    sample1['SPI'].value_counts()

    sample_size_per_spi = 10
    sampled_data = []
    for spi_value in range(1, 11):
        spi_samples = df[df['WPI'] == spi_value].sample(sample_size_per_spi, replace=True)
        sampled_data.append(spi_samples)

    sample2 = pd.concat(sampled_data, ignore_index=True)
    sample2 = sample2.sample(frac=1).reset_index(drop=True)
    sample2['WPI'].value_counts()


    features1 = list(features1)
    features1.remove("SPI")

    features2 = list(features2)
    features2.remove("WPI")

    fig = px.parallel_coordinates(sample1[features1], color="WPI", labels={"WPI": "WPI"},
        color_continuous_scale=px.colors.diverging.Tropic,
        color_continuous_midpoint=1)
    
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.parallel_coordinates(sample2[features2], color="SPI", labels={"SPI": "SPI"},
        color_continuous_scale=px.colors.diverging.Temps,
        color_continuous_midpoint=1)
    
    st.plotly_chart(fig2, use_container_width=True)

    # return fig

def parallelcoordinates(df):
    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Parallel Coordinates")
    with st.expander("Show code"):
        st.code("Code")
    
    districts = df["name"].unique()
    col1, col2 = st.columns(2)

    with col1:   
        features1 = st.multiselect("Select Features for SPI:", list(numcols), list(numcols[15:])) 
    with col2:   
        features2 = st.multiselect("Select Features for WPI:", list(numcols), list(numcols[15:])) 

    if features1 and features2:
        parallelCoord(df[numcols], features1, features2)