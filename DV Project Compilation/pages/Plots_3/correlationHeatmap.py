import streamlit as st
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

# This function is the one that returns the plot
def heatmap(df, features, featureX, colors):

    fig = go.Figure(data=go.Heatmap(
        z=df.corr().values, 
        x=features if featureX is None else [featureX], 
        y=features,
        colorscale='Viridis',
        text=df.corr().values,
        texttemplate="%{text:.2f}", 
        textfont={"size": 12} 
    ))

    fig.update_layout(
        title='Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        height = len(features) * 100
    )

    return fig

def correlationHeatmap(df):
    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Correlation Heatmap")
    with st.expander("Show code"):
        st.code("Code")
    
    districts = df["name"].unique()
    col1, col2 = st.columns(2)

    features = st.multiselect("Select Features:", list(numcols), list(numcols[15:])) 

    with col1:   
        temp = list(numcols)
        temp.append('None')
        
        featureX = st.selectbox("Select Feature", temp, index=len(temp) - 1,key="sf1")
        if featureX == 'None': featureX = None
    with col2:
        colorScale = st.selectbox("Select Color Scale", pc.PLOTLY_SCALES.keys(), index = 11)

    if features and colorScale:
        st.plotly_chart(heatmap(df[numcols], features, featureX, colorScale), use_container_width=True)