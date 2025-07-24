import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import plotly.colors as pc
import plotly.graph_objects as go

from pages.Plots_2.regression import regression
from pages.Plots_2.jointplot import jointplot

def main():
    st.set_page_config(layout="wide")
    st.subheader("Tamil Nadu: Weather Dashboard ")
    st.title("Bivariate Analysis")

    df = pd.read_parquet("pages\\Master.parquet", engine="fastparquet")

    df['datetime'] = pd.to_datetime(df['datetime'])
    numcols = df.select_dtypes(include=['number']).columns

    option = st.selectbox(
                "Select the type of plot:",
                ("Regression Plot","Joint Plot","Dataset"),
                index=None,
                placeholder="Select Plot...",
            )

    if option == "Regression Plot":
        regression(df)
    elif option == "Joint Plot":
        jointplot(df)
    else:
        st.write("### Dataset")
        st.dataframe(df)

if __name__ == "__main__":
    main()
