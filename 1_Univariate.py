import streamlit as st
import pandas as pd
import streamlit as st
import numpy as np

from pages.Plots_1.heatmap import timeseriesheatmap
from pages.Plots_1.chrolopleth import choloroplethmapbox
from pages.Plots_1.facetedColumn import facetedColumn
from pages.Plots_1.seasonalScatter import seasonalScatter
from pages.Plots_1.distribution import distribution
from pages.Plots_1.trends import plot_weather_trends

def main():
    st.set_page_config(layout="wide")
    st.subheader("Tamil Nadu: Weather Dashboard ")
    st.title("Univariate Analysis")

    df = pd.read_parquet("pages\\Master.parquet", engine="fastparquet")
    df['datetime'] = pd.to_datetime(df['datetime'])
    numcols = df.select_dtypes(include=['number']).columns

    option = st.selectbox(
                "Select the type of plot:",
                ("Chrolopleth Mapbox", "Faceted Area", "Seasonal Decomposition","Heatmap Plot", "Distribution Plot", "Trend Plots","Dataset"),
                index=None,
                placeholder="Select Plot...",
            )

    if option == "Chrolopleth Mapbox":
        choloroplethmapbox(df)
    
    elif option == "Faceted Area":
        facetedColumn(df)

    elif option == "Seasonal Decomposition":
        seasonalScatter(df)
    
    elif option == "Distribution Plot":
        distribution(df)

    elif option == "Trend Plots":
        plot_weather_trends(df)

    elif option == "Heatmap Plot":
        timeseriesheatmap(df)

    else:
        st.write("### Dataset")
        st.dataframe(df)

if __name__ == "__main__":
    main()