import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import plotly.colors as pc
import plotly.graph_objects as go

from pages.Plots_3.correlationHeatmap import correlationHeatmap
from pages.Plots_3.parallelCoor import parallelcoordinates
from pages.Plots_3.Dendrogram import Dendrogram

def main():
    st.set_page_config(layout="wide")
    st.subheader("Tamil Nadu: Weather Dashboard ")
    st.title("Multivariate Analysis")

    df = pd.read_parquet("pages\\Master.parquet", engine="fastparquet")

    df['datetime'] = pd.to_datetime(df['datetime'])
    numcols = df.select_dtypes(include=['number']).columns

    option = st.selectbox(
                "Select the type of plot:",
                ("Correlation Heatmap","Dataset", "Parallel Coordinates","Dendrogram"),
                index=None,
                placeholder="Select Plot...",
            )

    if option == "Correlation Heatmap":
        correlationHeatmap(df)
    elif option == "Parallel Coordinates":
        parallelcoordinates(df)
    elif option == "Dendrogram":
        Dendrogram(df)
    else:
        st.write("### Dataset")
        st.dataframe(df)



if __name__ == "__main__":
    main()
