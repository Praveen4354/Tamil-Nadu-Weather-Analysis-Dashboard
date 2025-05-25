# Tamil Nadu Weather Analysis Dashboard

## Project Overview
The **Tamil Nadu Weather Analysis Dashboard** is a Streamlit-based web application designed to explore, visualize, and predict weather patterns across Tamil Nadu districts using historical weather data. The project leverages the `Master3.csv` dataset, which contains detailed meteorological data (e.g., temperature, precipitation, humidity, wind speed) for districts. The dashboard provides interactive visualizations and predictive analytics, including univariate, bivariate, multivariate, and aggregated analyses, along with machine learning models for predicting the Standardized Precipitation Index (SPI).

The project aims to:
- Analyze weather trends across Tamil Nadu districts.
- Visualize spatial and temporal weather patterns using charts and maps.
- Predict drought-related indices (SPI) using machine learning models.
- Provide an intuitive interface for researchers, meteorologists, or enthusiasts to explore weather data.

## Dataset
The primary dataset is `Master3.csv`, which includes daily weather observations for Tamil Nadu districts. A sample for Ariyalur (Jan 1–11, 2021) includes:
- **Columns**: `name`, `datetime`, `tempmax`, `tempmin`, `temp`, `feelslikemax`, `feelslikemin`, `humidity`, `precip`, `precipprob`, `precipcover`, `preciptype`, `windgust`, `windspeed`, `windspeedmax`, `windspeedmean`, `windspeedmin`, `winddir`, `sealevelpressure`, `cloudcover`, `visibility`, `solarradiation`, `solarenergy`, `uvindex`, `sunrise`, `sunset`, `conditions`, `description`, `icon`.
- **Example Metrics**:
  - Temperature (max, min, mean, feels-like).
  - Precipitation (amount, probability, coverage, type).
  - Humidity, wind speed, cloud cover, UV index, etc.
- Additional data: `TN36.geojson` provides geospatial boundaries for Tamil Nadu’s 36 districts, used for choropleth maps.

The data was sourced via an API and processed into `Master3.csv` and `output_file.parquet` for efficient storage and analysis.

## Project Structure
The project is organized as follows:

```
DV Project Compilation/
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── pages/
│   ├── 1_Univariate.py           # Univariate analysis (e.g., missing values, distributions)
│   ├── 2_Bivariate.py            # Bivariate analysis (e.g., scatter plots, regression)
│   ├── 3_Multivariate.py         # Multivariate analysis (e.g., correlation heatmaps)
│   ├── 4_Aggregations.py         # Aggregated visualizations (e.g., bar charts, bubble plots)
│   ├── 5_Prediction.py           # SPI prediction using ML models
│   ├── Master.parquet            # Parquet version of dataset
│   ├── Plots_0/
│   │   └── missingValues.py      # Missing data analysis
│   ├── Plots_1/
│   │   ├── calendar.py           # Calendar heatmap
│   │   ├── chrolopleth.py        # Choropleth map for districts
│   │   ├── distribution.py       # Distribution plots
│   │   ├── facetedColumn.py      # Faceted column charts
│   │   ├── heatmap.py            # Heatmap for variable relationships
│   │   ├── seasonalScatter.py    # Seasonal scatter plots
│   │   ├── trends.py             # Trend line visualizations
│   ├── Plots_2/
│   │   ├── jointplot.py          # Joint plots for bivariate analysis
│   │   ├── regression.py         # Regression plots
│   ├── Plots_3/
│   │   ├── correlationHeatmap.py # Correlation heatmap
│   │   ├── Dendrogram.py         # Dendrogram for clustering
│   │   ├── parallelCoor.py       # Parallel coordinates plot
│   ├── Plots_4/
│   │   ├── barchart.py           # Bar charts for aggregations
│   │   ├── barpolar.py           # Polar bar charts
│   │   ├── districtBubble.py     # Bubble plots for district-level data
│   ├── Plots_5/
│   │   ├── models.py             # ML model visualizations
├── Dataset.py                    # Data loading and preprocessing
├── K-Nearest_Neighbors_SPI.pkl   # KNN model for SPI prediction
├── Logistic_Regression_SPI.pkl   # Logistic Regression model for SPI
├── Random_Forest_SPI.pkl         # Random Forest model for SPI
├── Support_Vector_Machine_SPI.pkl# SVM model for SPI
├── Master3.csv                   # Primary weather dataset
├── output_file.parquet           # Parquet output for processed data
├── TN36.geojson                  # Geospatial data for Tamil Nadu districts
```

## Features
The dashboard is divided into five main sections, accessible via Streamlit pages:

1. **Univariate Analysis** (`1_Univariate.py`, `Plots_0`):
   - Visualizes distributions of individual variables (e.g., temperature, precipitation).
   - Checks for missing values (`missingValues.py`).

2. **Bivariate Analysis** (`2_Bivariate.py`, `Plots_1`):
   - Explores relationships between two variables using:
     - Calendar heatmaps (`calendar.py`).
     - Choropleth maps for district-level data (`chrolopleth.py`).
     - Distribution plots (`distribution.py`).
     - Faceted column charts (`facetedColumn.py`).
     - Heatmaps (`heatmap.py`).
     - Seasonal scatter plots (`seasonalScatter.py`).
     - Trend lines (`trends.py`).

3. **Multivariate Analysis** (`3_Multivariate.py`, `Plots_2`):
   - Analyzes multiple variables using:
     - Correlation heatmaps (`correlationHeatmap.py`).
     - Dendrograms for clustering (`Dendrogram.py`).
     - Parallel coordinates plots (`parallelCoor.py`).
     - Joint plots (`jointplot.py`).
     - Regression plots (`regression.py`).

4. **Aggregations** (`4_Aggregations.py`, `Plots_3`):
   - Summarizes data with:
     - Bar charts (`barchart.py`).
     - Polar bar charts (`barpolar.py`).
     - District-level bubble plots (`districtBubble.py`).

5. **Prediction** (`5_Prediction.py`, `Plots_4`):
   - Displays SPI predictions using four machine learning models:
     - K-Nearest Neighbors (`K-Nearest_Neighbors_SPI.pkl`).
     - Logistic Regression (`Logistic_Regression_SPI.pkl`).
     - Random Forest (`Random_Forest_SPI.pkl`).
     - Support Vector Machine (`Support_Vector_Machine_SPI.pkl`).
   - Visualizes model outputs (`models.py`).

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: For building the interactive web dashboard.
- **Pandas**: Data manipulation and preprocessing.
- **Plotly/Seaborn/Matplotlib**: For visualizations (inferred from plot types).
- **Scikit-learn**: For machine learning models (KNN, Logistic Regression, Random Forest, SVM).
- **GeoPandas/Folium**: For geospatial analysis with `TN36.geojson` (inferred for choropleth maps).
- **Parquet**: Efficient data storage (`Master.parquet`, `output_file.parquet`).

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd DV-Project-Compilation
   ```

2. **Install Dependencies**:
   Ensure Python 3.11 is installed. Install required packages:
   ```bash
   pip install streamlit pandas plotly seaborn matplotlib scikit-learn geopandas folium
   ```

3. **Run the Dashboard**:
   Start the Streamlit app:
   ```bash
   streamlit run Dataset.py
   ```
   - This launches the dashboard, with navigation to the five analysis pages.
   - Ensure `Master3.csv`, `TN36.geojson`, and pickle files are in the project directory.

4. **Explore the Dashboard**:
   - Access the app in your browser (typically `http://localhost:8501`).
   - Navigate through the pages (Univariate, Bivariate, Multivariate, Aggregations, Prediction) using the sidebar.
   - Interact with visualizations and filters (e.g., select districts, date ranges).

## Usage
- **Univariate**: Check distributions and missing data to understand individual weather metrics.
- **Bivariate**: Explore relationships (e.g., temperature vs. precipitation) to identify patterns.
- **Multivariate**: Analyze complex interactions between multiple variables.
- **Aggregations**: View summarized data (e.g., average precipitation by district).
- **Prediction**: Input weather parameters to predict SPI and assess drought risk.

## Example Visualizations
- **Choropleth Map**: Displays precipitation or SPI across Tamil Nadu districts.
- **Calendar Heatmap**: Shows daily temperature or precipitation trends.
- **Correlation Heatmap**: Highlights relationships between variables like temperature, humidity, and wind speed.
- **SPI Prediction**: Visualizes predicted drought indices using ML models.

## Future Improvements
- Add real-time API integration for live weather data.
- Enhance interactivity with more filters (e.g., time periods, specific weather conditions).
- Include additional ML models or hyperparameter tuning for improved SPI predictions.
- Optimize performance for large datasets using caching or database integration.

## License
This project is licensed under the MIT License.
