import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import plotly.colors as pc
import plotly.graph_objects as go

from pages.Plots_5.models import model_prediction_and_evaluation

def main():
    st.set_page_config(layout="wide")
    st.subheader("Tamil Nadu: Weather Dashboard ")
    st.title("Predictive Modeling")

    df = pd.read_parquet("pages\\Master.parquet", engine="fastparquet")

    df['datetime'] = pd.to_datetime(df['datetime'])

    option = st.selectbox(
                "Select the type of plot:",
                ("Dataset", "Predictive Modeling"),
                index=None,
                placeholder="Select Plot...",
            )

    if option == "Predictive Modeling":
        model_prediction_and_evaluation(df)
    else:
        st.write("### Dataset")
        st.dataframe(df)

if __name__ == "__main__":
    main()
