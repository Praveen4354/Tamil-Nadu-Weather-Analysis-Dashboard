import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import plotly.colors as pc
import plotly.graph_objects as go

from pages.Plots_0.missingValues import missingValues

def main():
    st.set_page_config(layout="wide")
    st.title("Tamil Nadu: Weather Dashboard")

    df = pd.read_csv("Master 2021 - 23.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    option = st.selectbox(
                "Select the type of plot:",
                ("Missing Values","Dataset"),
                index=None,
                placeholder="Select Plot...",
            )

    if option == "Missing Values":
        missingValues(df)
    else:
        st.write("### Dataset")
        st.dataframe(df)
        st.subheader("Basic Statistics")
        st.write(df.describe())

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Features")
            st.table(pd.DataFrame(zip(df.columns.values, df.dtypes), columns=['Feature','Data Type']))
        with col2:
            st.subheader("Districts")
            st.table(pd.DataFrame(df['name'].unique(),columns=['Districts of TN']))
        with col3:
            st.subheader("Categorical Columns Value Counts")
        
            for col in ['icon','description','conditions']:
                st.write(f"Value Counts of {col}")
                st.bar_chart(df[col].value_counts())

if __name__ == "__main__":
    main()
