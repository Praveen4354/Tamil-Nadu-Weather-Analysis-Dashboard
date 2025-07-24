import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.colors as pc

def district_bubble_plot(df):
    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Heatmap Plot")
    with st.expander("Show code"):
        st.code("Code")
    
    districts = df["name"].unique()

    col1, col2 = st.columns(2)

    dists = st.multiselect("Select Districts:",districts, districts) 

    with col1:   
        featureX = st.selectbox("Select Feature", numcols, key="sf1")
    with col2:
        colorScale = st.selectbox("Select Color Scale", pc.PLOTLY_SCALES.keys(), index = 6)
        
    df1 = df[df['name'].isin(dists)]
        
    district_avg = df1.groupby('name')[featureX].mean().reset_index()
    

    fig = px.scatter(district_avg, x='name', y=featureX, size=featureX, color=featureX, color_continuous_scale = colorScale,
                     title=f'District-wise Average {featureX}')
    
    st.plotly_chart(fig, use_container_width=True)