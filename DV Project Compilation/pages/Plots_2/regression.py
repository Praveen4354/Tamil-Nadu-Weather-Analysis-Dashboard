import pandas as pd
import plotly.express as px
import streamlit as st

def regression(df):
    numcols = df.select_dtypes(include=['number']).columns
    st.write("### Regression Plot")
    with st.expander("Show code"):
        st.code("""
            import pandas as pd
            import plotly.express as px

            df = pd.read_csv("your_file.csv")
            fig = px.scatter(df, x='X_column', y='Y_column')
            st.plotly_chart(fig)
        """, language="python")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_district = st.selectbox("Select District", df['name'].unique())
    with col2:   
        selected_featureX = st.selectbox("Select X Feature", numcols)
    with col3:
        selected_featureY = st.selectbox("Select Y Feature", numcols)
    with col4:
        min_date = pd.to_datetime(df['datetime']).min()
        max_date = pd.to_datetime(df['datetime']).max()
        dateRan1 = st.date_input("Select Date Range", 
                                            min_value=min_date, max_value=max_date,
                                            value=(min_date, max_date))

    if selected_district and selected_featureX and selected_featureY:
        df1 = df[df['name'] == selected_district]

        fdf = df1[(df1['datetime'] >= pd.to_datetime(dateRan1[0])) & (df1['datetime'] <= pd.to_datetime(dateRan1[1]))]

        fig = px.scatter(
            fdf, 
            x=selected_featureX, 
            y=selected_featureY, 
            trendline="ols", 
            trendline_color_override='white', 
            color="SPI", 
            color_discrete_sequence=px.colors.qualitative.Pastel1
        )

        fig.update_layout(height = 600)
        st.plotly_chart(fig, use_container_width=True)