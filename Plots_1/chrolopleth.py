import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import plotly.colors as pc
import plotly.graph_objects as go

def modifiedGDF(master, gdf, dts, colN, swap = False, text = ""):
    if not text: text = colN
    for idx, row in gdf.iterrows():
        district = row['District']
        ind = gdf.index[gdf['District'] == district][0]
        if not swap:
            gdf.loc[ind, 'view'] = master.loc[(master['name'] == district) & (master['datetime'] == dts), colN].values[0]
        else:
            gdf.loc[ind, 'view'] = master.loc[(master['name'] == district), colN].values[0]

    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry.__geo_interface__,
        locations=gdf.index,
        color='view', 
        center={"lat": 11.127123, "lon": 78.656891},
        zoom=6,
        hover_name='District',
        mapbox_style = "carto-positron"
    )

    fig.update_layout(
        geo=dict(bgcolor= 'rgba(0,0,0,0)'),
        height = 800,
        annotations=[
            dict(
                x=0.5,
                y=0,
                xref="paper",
                yref="paper",
                text=text,
                showarrow=False,
                font=dict(
                    family="Bahnschrift",
                    size=20,
                    color="black"
                )
            )
        ],
    )

    return fig

def choloroplethmapbox(df):
    
    numcols = df.select_dtypes(include=['number']).columns
    gdf = gpd.read_file('TN36.geojson')

    st.write("### Chrolopleth Mapbox")
    with st.expander("Show code"):
        st.code("""
            import pandas as pd
            import plotly.express as px

            df = pd.read_csv("your_file.csv")
            fig = px.scatter(df, x='X_column', y='Y_column')
            st.plotly_chart(fig)
        """, language="python")

    col1, col2 = st.columns(2)
    with col1: 
        X1 = st.selectbox("Select Feature", numcols)
    with col2:
        dts = pd.to_datetime('2023-01-01')
        date1 = st.date_input("Select Date", dts)

    if X1 and date1:

        fig = modifiedGDF(df.copy(), gdf, date1, X1,swap=True)
        st.plotly_chart(fig, use_container_width=True)