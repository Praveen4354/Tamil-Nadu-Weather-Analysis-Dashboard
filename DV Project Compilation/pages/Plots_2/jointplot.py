import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.colors as pc

def jointplot(df):
    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Heatmap Plot")
    with st.expander("Show code"):
        st.code("Code")
    
    districts = df["name"].unique()

    col1, col2, col3 = st.columns(3)

    with col1:   
        featureX = st.selectbox("Select Feature", numcols, key="sf1")
    with col2:   
        featureY = st.selectbox("Select Feature", numcols, key="sf2")
    with col3:
        colorScale = st.selectbox("Select Color Scale", pc.PLOTLY_SCALES.keys(), index = 6)

    col4, col5, col6 = st.columns(3)

    with col4:
        plot = st.selectbox("Select Plot Type", ['Density Heatmap', 'Density Contour'], key = 'sb4', index=1)
    with col5:
        marginx = st.selectbox("Select Marginal X:", ["histogram", "rug", "box", "violin", None], key = 'sb5')
    with col6:
        marginy = st.selectbox("Select Marginal Y:", ["histogram", "rug", "box", "violin", None], key = 'sb6')
    
    if plot == 'Density Heatmap':

        fig = px.density_heatmap(data_frame=df, x=featureX, y=featureY, title="Density Heatmap", labels={'x': featureX, 'y': featureY},
                             marginal_x=marginx, marginal_y=marginy, nbinsx=50, nbinsy=50, histfunc='avg')
    else:

        fig = px.density_contour(data_frame=df, x=featureX, y=featureY, title="Density Heatmap", labels={'x': featureX, 'y': featureY},
                             marginal_x=marginx, marginal_y=marginy, nbinsx=50, nbinsy=50, histfunc='avg')
        
    fig.update_layout(height = 800)
    
    st.plotly_chart(fig, use_container_width=True)