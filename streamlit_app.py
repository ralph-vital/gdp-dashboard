import streamlit as st
import pandas as pd
import math
from pathlib import Path
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='HAITI Edution Indicator dashboard',
    # This is an emoji shortcode. Could be a URL too.
    page_icon=':earth_americas:',
)

# -----------------------------------------------------------------------------
# Declare some useful functions.


@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/exchange-rates_hti.csv'
    btc = pd.read_csv(DATA_FILENAME, skiprows=[1,])

    MIN_YEAR = 1970
    MAX_YEAR = 2023

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    # gdp_df = raw_gdp_df.melt(
    #     ['Country Code'],
    #     [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
    #     'Year',
    #     'GDP',
    # )

    btc_x = btc[['StartDate', 'Value']]
    btc_x = btc_x.sort_values('StartDate')
    btc_x.index = pd.to_datetime(btc_x['StartDate'], format='%Y-%m-%d')
    del btc_x['StartDate']

    # Convert years from string to integers

    return btc_x


# -----------------------------------------------------------------------------
# Draw the actual page
# Set the title that appears at the top of the page.
'''
# Forcasting Haiti Xchange Rate using SARIMEX


SARIMAX and ARIMA forecasters
SARIMAX (Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors) is a generalization of the ARIMA model that incorporates both seasonality and exogenous variables. SARIMAX models are among the most widely used statistical forecasting models with excellent forecasting performance.

In the SARIMAX model notation, the parameters  
p ,  d , and  q represent the autoregressive, differencing, and moving-average components, respectively.  
P,  D, and  Q denote the same components for the seasonal part of the model, with  m representing the number of periods in each season.

p: is the order (number of time lags) of the autoregressive part of the model.

d: is the degree of differencing (the number of times that past values have been subtracted from the data).

q: is the order of the moving-average part of the model.

P: is the order (number of time lags) of the seasonal part of the model

D: is the degree of differencing (the number of times the data have had past values subtracted) of the seasonal part of the model.

Q: is the order of the moving-average of the seasonal part of the model.

m: refers to the number of periods in each season.
'''

# Add some spacing


@st.cache_data
def get_y(btc_x):
    train = btc_x[btc_x.index < pd.to_datetime(
        "2021-12-31", format='%Y-%m-%d')]
    test = btc_x[btc_x.index > pd.to_datetime(
        "2021-12-31", format='%Y-%m-%d')]
    return train, test


# get the data
gdp_df = get_gdp_data()
y, test = get_y(gdp_df)

st.sidebar.header('SARIMA Parameters:')
p = st.sidebar.select_slider(
    "Select p",
    options=[
        0, 1, 2, 3, 4, 5, 6],
    value=1
)

d = st.sidebar.select_slider(
    "Select d",
    options=[
        0, 1, 2, 3, 4, 5, 6],
    value=0
)

q = st.sidebar.select_slider(
    "Select q",
    options=[
        0, 1, 2, 3, 4, 5, 6],
    value=1)
P = st.sidebar.select_slider(
    "Select P",
    options=[
        0, 1, 2, 3, 4, 5, 6],
    value=0
)
D = st.sidebar.select_slider(
    "Select D",
    options=[
        0, 1, 2, 3, 4, 5, 6],
    value=0
)
Q = st.sidebar.select_slider(
    "Select Q",
    options=[
        0, 1, 2, 3, 4, 5, 6],
    value=0
)

m = st.sidebar.select_slider(
    "Select m",
    options=[
        0, 1, 2, 3, 4, 5, 6],
    value=0
)


def get_fit(SARIMAXmodel):
    SARIMAXmodel = SARIMAXmodel.fit()
    return SARIMAXmodel


SARIMAXmodel = SARIMAX(y, order=(5, 4, 2), seasonal_order=(2, 2, 2, 12))
with st.spinner('Wait for it... it might take a few minutes'):
    SARIMAXmodel = get_fit(SARIMAXmodel)


y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha=0.05)
y_pred_df["Predictions"] = SARIMAXmodel.predict(
    start=y_pred_df.index[0], end=y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"]



st.header("Forcasting Haiti Xchange US Dollar/Gourdes")


gdf_x = gdp_df.reset_index()
gdf_x = gdf_x.sort_values('StartDate')

gdf_x['StartDate'] = pd.to_datetime(gdf_x['StartDate'], format='%Y-%m-%d')
y_pred_out = y_pred_out.reset_index()
print(y_pred_out)

fig_x = px.line(gdf_x, x=gdf_x.StartDate, y=gdf_x.Value)

fig_y = go.Figure()

fig_y.add_trace(go.Scatter(x=gdf_x.StartDate,
                y=gdf_x.Value,  name='True Values', line=dict(color='red', width=4)))
fig_y.add_trace(go.Scatter(x=y_pred_out.StartDate, y=y_pred_out.Predictions, name='Predicted Values',
                           line=dict(color='royalblue', width=4)))
fig_y.update_layout(title='Forcasting Haiti Xchage US Dollar/Gourdes',
                    xaxis_title='Date',
                    yaxis_title='Value')
st.plotly_chart(fig_y)
# cols = st.columns(4)
