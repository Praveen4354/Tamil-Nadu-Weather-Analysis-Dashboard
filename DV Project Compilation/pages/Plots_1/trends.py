import pandas as pd
import plotly.express as px
import streamlit as st

def plot_weather_trends(data):

    data['datetime'] = pd.to_datetime(data['datetime'])
    
    st.title('Weather Trends Analysis')

    col1, col2 = st.columns(2)
    
    with col1:
        # District and Year dropdowns
        district_names = data['name'].unique().tolist()
        district_name = st.selectbox('Select District', district_names)
    
    with col2:
        years = data['datetime'].dt.year.unique().tolist()
        year = st.selectbox('Select Year', years)

    # Filter data based on selections
    filtered_data = data[(data['name'] == district_name) & (data['datetime'].dt.year == year)]
    
    def create_line_plot(filtered_data, x_col, y_cols, title, yaxis_title):
        title = ""
        fig = px.line(filtered_data, x=x_col, y=y_cols, title=title)
        fig.update_layout(yaxis_title=yaxis_title)
        st.plotly_chart(fig, use_container_width=True)
    
    # Temperature trends
    st.subheader(f'Temperature Trends in {district_name} ({year})')
    create_line_plot(filtered_data, 'datetime', ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin'], yaxis_title= 'Temperature (°C)', title="")

    # Humidity and precipitation trends
    st.subheader(f'Humidity and Precipitation Trends in {district_name} ({year})')
    create_line_plot(filtered_data, 'datetime', ['humidity', 'precip'], 
                     f'Humidity and Precipitation Trends in {district_name} ({year})', 'Humidity (%) / Precipitation (mm)')

    # Solar radiation and UV index trends
    st.subheader(f'Solar Radiation and UV Index Trends in {district_name} ({year})')
    create_line_plot(filtered_data, 'datetime', ['solarradiation', 'uvindex'], 
                     f'Solar Radiation and UV Index Trends in {district_name} ({year})', 'Solar Radiation (W/m²) / UV Index')

    # Wind speed and direction
    st.subheader(f'Wind Speed and Direction Trends in {district_name} ({year})')
    create_line_plot(filtered_data, 'datetime', ['windspeed', 'winddir'], 
                     f'Wind Speed and Direction Trends in {district_name} ({year})', 'Wind Speed (km/h) / Wind Direction (°)')

    # Sea level pressure
    st.subheader(f'Sea Level Pressure Trends in {district_name} ({year})')
    create_line_plot(filtered_data, 'datetime', 'sealevelpressure', 
                     f'Sea Level Pressure Trends in {district_name} ({year})', 'Sea Level Pressure (hPa)')

    # Cloud cover
    st.subheader(f'Cloud Cover Trends in {district_name} ({year})')
    create_line_plot(filtered_data, 'datetime', 'cloudcover', 
                     f'Cloud Cover Trends in {district_name} ({year})', 'Cloud Cover (%)')

    # Visibility trends
    st.subheader(f'Visibility Trends in {district_name} ({year})')
    create_line_plot(filtered_data, 'datetime', 'visibility', 
                     f'Visibility Trends in {district_name} ({year})', 'Visibility (km)')

    # Wind gust trends
    st.subheader(f'Wind Gust Trends in {district_name} ({year})')
    create_line_plot(filtered_data, 'datetime', 'windgust', 
                     f'Wind Gust Trends in {district_name} ({year})', 'Wind Gust (km/h)')

    # SPI and WPI trends
    st.subheader(f'SPI and WPI Trends in {district_name} ({year})')
    create_line_plot(filtered_data, 'datetime', ['SPI', 'WPI'], 
                     f'SPI and WPI Trends in {district_name} ({year})', 'Index Value')