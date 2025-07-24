import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


def plot_missing_values(df, row_range, columns=['preciptype', 'windgust', 'sealevelpressure']):

    start_row, end_row = row_range
    df_subset = df.iloc[start_row:end_row]
    
    mdf = df_subset[columns]

    fig = px.imshow(mdf.isnull(), 
                    labels=dict(color="Missing Values"),
                    title="Missing Values Matrix for Selected Columns",
                    width=500, height=1000,
                    color_continuous_scale=[(0, "white"), (1, "black")]) 

    fig.update_layout(xaxis_title="Columns",
                      yaxis_title="Index",
                      xaxis_showgrid=False,
                      yaxis_showgrid=False,
                      yaxis_autorange='reversed')
    return fig

def missingValues(df):
    st.write("### Chrolopleth Mapbox")
    with st.expander("Show code"):
        st.code("code", language="python")
    
    row, col = df.shape

    row_range = st.select_slider("Select Row Range: ", options=range(1, row), value=(1, 1095))
    
    st.plotly_chart(plot_missing_values(df, row_range=row_range), use_container_width=True)

    st.write("Windgust is closely related to windspeed, windspeedmean. So, We can apply regression techniques to fill missing values since missing values appears to be present as continuous patches.")
    st.markdown("""- Split the dataset into two parts: one with non-missing values in the 'windgust' column (training data) and the other with missing values (testing data).
- Train a linear regression model on the training data, where 'windspeedmean' is the independent variable and 'windgust' is the dependent variable.
- Use the trained model to predict the missing values of 'windgust' in the testing data.
- Replace the missing values in the original dataset with the predicted values.""")
    
    st.code("""def impute_missing_windgust(df):
    train_data = df.dropna(subset=['windgust'])
    test_data = df[df['windgust'].isnull()]

    X_train = train_data[['windspeedmean']]
    y_train = train_data['windgust']
    X_test = test_data[['windspeedmean']]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predicted_windgust = model.predict(X_test)

    df.loc[df['windgust'].isnull(), 'windgust'] = predicted_windgust

    return df

sample = impute_missing_windgust(df[['windgust','windspeed','windspeedmean']])
df['windgust'] = sample['windgust']""")

    st.write("Inference : Single Missing values can be imputed by taking aggregate mean of the neibours")

    st.code("df[['name', 'datetime' ,'visibility']][df['visibility'].isnull()]['name'].value_counts()")
    sam = df[['name', 'datetime' ,'visibility']][df['visibility'].isnull()]['name'].value_counts()
    st.table(sam)

    st.write("Missing values that are spread out and minimal in other districts are filled by taking neighbour mean for head(3) and tail(3).")

    st.code("""for x in range(len(datesMissing)):
    aW = 76
    bW = 36
    cW = 3
    dW = 123
    eW = 87

    a = df.loc[(df['name'] == 'Salem') & (df['datetime'] == datesMissing['datetime'].iloc[x]), 'visibility'].values[0]
    b = df.loc[(df['name'] == 'Viluppuram') & (df['datetime'] == datesMissing['datetime'].iloc[x]), 'visibility'].values[0]
    c = df.loc[(df['name'] == 'Perambalur') & (df['datetime'] == datesMissing['datetime'].iloc[x]), 'visibility'].values[0]
    d = df.loc[(df['name'] == 'Tiruvannamalai') & (df['datetime'] == datesMissing['datetime'].iloc[x]), 'visibility'].values[0]
    e = df.loc[(df['name'] == 'Cuddalore') & (df['datetime'] == datesMissing['datetime'].iloc[x]), 'visibility'].values[0]

    weighted_average = (a * aW + b * bW + c * cW + d * dW + e * eW) / (aW + bW + cW + dW + eW)

    datesMissing.loc[x, 'visibility'] = weighted_average.round(4)""")

    st.write("Proposition: Continuous patch of missing values in 'Kallakurichi' can be imputed by taking a weighted average of neighbouring districts's visibility value per day.")

