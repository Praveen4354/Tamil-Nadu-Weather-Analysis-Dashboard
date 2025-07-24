import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import plotly.colors as pc
import plotly.graph_objects as go

from pages.Plots_4.barpolar import barpolar
from pages.Plots_4.barchart import barchart
from pages.Plots_4.districtBubble import district_bubble_plot

def main():
    st.set_page_config(layout="wide")
    st.subheader("Tamil Nadu: Weather Dashboard ")
    st.title("Aggregations and Transformations")

    df = pd.read_parquet("pages\\Master.parquet", engine="fastparquet")

    df['datetime'] = pd.to_datetime(df['datetime'])

    option = st.selectbox(
                "Select the type of plot:",
                ("Dataset", "Barpolar", "Barchart", "District Wise Bubble Plot"),
                index=None,
                placeholder="Select Plot...",
            )

    if option == "Barpolar":
        barpolar(df)
    elif option == "Barchart":
        barchart(df)
    elif option == "District Wise Bubble Plot":
        district_bubble_plot(df)
    else:
        st.write("### Dataset")
        st.dataframe(df)

if __name__ == "__main__":
    main()
