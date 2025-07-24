import streamlit as st
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.express as px

def facetedColumnPlot(df, districts, featureX, start, end):

    df1 = df[df['name'].isin(districts)]
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    dfs = df1[(df1['datetime'] >= start) & (df1['datetime'] <= end)]

    df = dfs.pivot(index='datetime', columns='name')[featureX]
    fig = px.area(df, facet_col="name", facet_col_wrap=2)
    fig.update_layout(title=f'Area Plot across districts for {featureX}', height = (len(districts)/2)*300)

    return fig

def facetedColumn(df):
    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Heatmap Plot")
    with st.expander("Show code"):
        st.code("Code")
    
    districts = df["name"].unique()
    col1, col2 = st.columns(2)

    dists = st.multiselect("Select Districts:",districts, districts[:6]) 

    with col1:   
        featureX = st.selectbox("Select Feature", numcols, key="sf1")
    with col2:
        min_date = pd.to_datetime(df['datetime']).min()
        max_date = pd.to_datetime(df['datetime']).max()
        temp_max_date = pd.to_datetime("2021-01-31")
        dateRan = st.date_input("Select Date Range", 
                                            min_value=min_date, max_value=max_date,
                                            value=(min_date, temp_max_date), key="sdr1")

    if featureX:
        st.plotly_chart(facetedColumnPlot(df, dists, featureX, dateRan[0], dateRan[1]), use_container_width=True)