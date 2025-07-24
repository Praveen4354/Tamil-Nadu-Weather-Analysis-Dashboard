import streamlit as st
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

# This function is the one that returns the plot
def heatmap(df, districts, featureX, colors, start, end):

    df1 = df[df['name'].isin(districts)]
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    dfs = df1[(df1['datetime'] >= start) & (df1['datetime'] <= end)]

    fig = go.Figure(data=go.Heatmap(
        z=dfs[featureX],
        x=dfs['datetime'],
        y=dfs['name'],
        colorscale=colors))

    fig.update_layout(
        title=f'Heapmap across districts for {featureX}',
        
        )

    return fig

def timeseriesheatmap(df):
    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Heatmap Plot")
    with st.expander("Show code"):
        st.code("Code")
    
    districts = df["name"].unique()
    col1, col2, col3 = st.columns(3)

    dists = st.multiselect("Select Districts:",districts, districts) 

    with col1:   
        featureX = st.selectbox("Select Feature", numcols, key="sf1")
    with col2:
        colorScale = st.selectbox("Select Color Scale", pc.PLOTLY_SCALES.keys(), index = 6)
    with col3:
        min_date = pd.to_datetime(df['datetime']).min()
        max_date = pd.to_datetime(df['datetime']).max()
        temp_max_date = pd.to_datetime("2021-01-31")
        dateRan = st.date_input("Select Date Range", 
                                            min_value=min_date, max_value=max_date,
                                            value=(min_date, temp_max_date), key="sdr1")

    if featureX and colorScale:
        st.plotly_chart(heatmap(df, dists, featureX, colorScale, dateRan[0], dateRan[1]), use_container_width=True)