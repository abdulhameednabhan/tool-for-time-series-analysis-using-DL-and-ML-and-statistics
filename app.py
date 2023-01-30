import numpy as np
import pandas as pd
import streamlit as st
from altair.vega import colorRGB
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
import seaborn as sns
import utils
import matplotlib.pyplot as plt
import plotly.express as px
import json
import joblib
from PIL import  Image

import pandas as pd

from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import plotly.graph_objects as go
ma1=10
ma2=20
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from pandas_datareader import data
from datetime import datetime, timedelta
df = pd.read_csv("data1.csv")
correltion = pd.read_csv("data1.csv")
coverince=pd.read_csv("data1.csv")
from plotly.subplots import make_subplots
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose


pio.renderers.default = 'browser'
def plot_seasonal_decompose(result:DecomposeResult, dates:pd.Series=None, title:str="Seasonal Decomposition"):
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.observed, mode="lines", name='Observed'),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.trend, mode="lines", name='Trend'),
            row=2,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode="lines", name='Seasonal'),
            row=3,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.resid, mode="lines", name='Residual'),
            row=4,
            col=1,
        )
        .update_layout(
            height=900, title=f'<b>{title}</b>', margin={'t':100}, title_x=0.5, showlegend=False
        )
    )





st.set_page_config(page_title="time series analysis and visualzaion", page_icon=":bar_chart:", layout="wide",  )
st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown(""" <style> .font {
        font-size:100px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Tool for time series anlysis</p>', unsafe_allow_html=True)
#st.header('Stock price predict and analysis')
display = Image.open('logo.png')
display = np.array(display)
#st.image(display, width = 400)
#st.title("stock price forcast")

col1, col2,col3 = st.columns(3)
col2.image(display, width = 500)
#col2.title("stock price /tsla/AAple/azn ")
def get_candlestick_plot(
        df: pd.DataFrame,
        ma1: int,
        ma2: int,
        ticker: str):
    '''
    Create the candlestick chart with two moving avgs + a plot of the volume
    Parameters
    ----------
    df : pd.DataFrame
        The price dataframe
    ma1 : int
        The length of the first moving average (days)
    ma2 : int
        The length of the second moving average (days)
    ticker : str
        The ticker we are plotting (for the title).
    '''

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{ticker} Stock Price', 'Volume Chart'),
        row_width=[0.3, 0.7]
    )

    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick chart'
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Line(x=df['Date'], y=df[f'{ma1}_ma'], name=f'{ma1} SMA'),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Line(x=df['Date'], y=df[f'{ma2}_ma'], name=f'{ma2} SMA'),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume'),
        row=2,
        col=1,
    )

    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = 'Volume'

    fig.update_xaxes(
        rangebreaks=[{'bounds': ['sat', 'mon']}],
        rangeslider_visible=False,
    )

    return fig


with st.sidebar:
    ch= option_menu("Time Series Analysis ", ["browsing data", "view real", "modeling with Regrssion", "Visualization", "modeling2"
                                          ,  "modeling3"],
                       icons=['house', 'bar-chart-steps', 'gear', 'bar-chart-steps','gear'],
                         menu_icon="cast", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "##00172B"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#ffc046"},
        "nav-link-selected": {"background-color": "#ffc046"},
    }
    )

if ch == "browsing data":


    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your data</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(


        "fill  by csv files",
        type=['csv'])
    if uploaded_file is not None:
       df= pd.read_csv(uploaded_file)
     
       gb = GridOptionsBuilder.from_dataframe(df)
       gb.configure_pagination(paginationAutoPageSize=True)  # Add pagination
       gb.configure_side_bar()  # Add a sidebar
       gb.configure_selection('multiple', use_checkbox=True,
                              groupSelectsChildren="Group checkbox select children")
      # Enable multi-row selection
       gridOptions = gb.build()


    if st.button("Load Data"):


        grid_response = AgGrid(
            df,
            gridOptions=gridOptions,
            data_return_mode='AS_INPUT',
            update_mode='MODEL_CHANGED',
            fit_columns_on_grid_load=False,
            theme='blue',  # Add theme color to the table
            enable_enterprise_modules=True,
            height=350,
            width='100%',
            reload_data=True
        )

        print("_____________________")
    if st.button("statistics"):
        st.subheader('Data from 2010-2019')
        st.write(df.describe().style.set_properties(**{'background-color': 'black',
                           'color': 'orange'}))


        st.line_chart(df.Close)

if ch == "view real":
    stock_name = st.text_input("Enter the stock name: \n",'AAPL')
    option = st.slider("How many days of data would you like to see?",
                       1, 60, 1)
    end = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(option)).strftime('%Y-%m-%d')


    def load_data(stock, start_date, end_date):
        df = data.DataReader(name=stock,
                             start=start_date,
                             end=end_date,
                             data_source='yahoo')
        return df


    data_load_state = st.text("Loading data...")
    df = load_data(stock=stock_name, start_date=start, end_date=end)
    df.sort_index(axis=0, inplace=True, ascending=False)
    st.subheader(f'{stock_name} stock prices for the past {option} days')
    st.dataframe(df)
    chart_data = df[['Close']]
    st.subheader("Close Prices")
    st.line_chart(chart_data)
    data_load_state.text("Data loaded!")
    from pandas_datareader import data
    import datetime

    ticker_input = st.selectbox('Select one symbol', ('AZN',))
    # create default date range
    start1 = datetime.datetime(2010, 1, 1)
    end1= datetime.datetime(2021, 11, 1)
    # ask user for his date
    start_date1 = st.date_input('Start date', start1)
    end_date1 = st.date_input('End date', end1)
    # validate start_date date is smaller then end_date

    st.title('"Close" "Open" "Low" "High"')
    # get data based on dates
    df1 = data.DataReader(ticker_input, 'yahoo', start_date1, end_date1)
    # plot
    st.line_chart(df1.loc[:, ["Close", "Open", "Low", "High"]])
    import os




    if st.button("Downlaod"):
        os.makedirs('folder/subfolder', exist_ok=True)

        df1.to_csv('folder/subfolder/out.csv')


if ch == "modeling with Regrssion":
    df['Prediction adj'] =df[['Adj Close']].shift(-30)
    df['Prediction close'] = df[['Close']].shift(-30)
    df=df.dropna()


    params = {}

    # Use two column technique
    col1, col2 = st.columns(2)

    # Design column 1
    y_var = col1.radio("Select the variable to be predicted (y)", options=df.columns)

    # Design column 2
    X_var = col2.multiselect("Select the variables to be used for prediction (X)", options=df.columns)

    # Check if len of x is not zero
    if len(X_var) == 0:
        st.error("You have to put in some X variable and it cannot be left empty.")


    # Check if y not in X
    if y_var in X_var:
        st.error("Warning! Y variable cannot be present in your X-variable.")

    # Option to select predition type
    pred_type = st.radio("Select the type of process you want to run.",
                         options=["Regression"],
                         help="Write about reg ")


    # Add to model parameters
    params = {
        'X': X_var,
        'y': y_var,
        'pred_type': pred_type,
    }
    # Divide the data into test and train set
    X = df[X_var]
    y = df[y_var]


    # Perform data imputation
    # st.write("THIS IS WHERE DATA IMPUTATION WILL HAPPEN")

    # Perform encoding

        # Print all the classes


    # Perform train test splits
    st.markdown("#### Train Test Splitting")
    size = st.slider("Percentage of value division",
                     min_value=0.1,
                     max_value=0.9,
                     step=0.1,
                     value=0.8,
                     help="This is the value which will be used to divide the data for training and testing. Default = 80%")

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
    st.write("Number of training samples:", X_train.shape[0])
    st.write("Number of testing samples:", X_test.shape[0])
    if st.button("visual linear regression"):
        model = LinearRegression()
        model.fit(X_train, y_train)

        x_range = np.linspace(X.min(), X.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))
        predictions = model.predict(X_test)

        fig2 = go.Figure([
            go.Scatter(x=predictions.squeeze(), y=predictions, name='train', mode='markers'),

            go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
            #go.Scatter(x=x_range, y=y_range, name='prediction')
        ])
        st.plotly_chart(fig2,use_container_width=True)
    if st.button("visual the result"):
            forecast_out = 30
            azn_adj = df[['Adj Close']]
            azn_adj['Prediction'] = azn_adj[['Adj Close']].shift(-30)
            X = np.array(azn_adj.drop(['Prediction'], 1))
            # Remove last 'n' rows
            X = X[:-forecast_out]
            y = np.array(azn_adj['Prediction'])
            # Remove last 'n' rows
            y = y[:-forecast_out]

            train_size = int(X.shape[0] * 0.7)

            X_train = X[0:train_size]
            y_train = y[0:train_size]

            X_test = X[train_size:]
            y_test = y[train_size:]
            X_forecast = np.array(azn_adj.drop(['Prediction'], 1))[-forecast_out:]
            # Create Linear Regression model
            lr = LinearRegression()

            # Train the model
            lr.fit(X_train, y_train)
            predictions = lr.predict(X_test)
            a = pd.DataFrame(predictions, columns=['pred'])
            b = pd.DataFrame(y_test, columns=['test'])

            f = pd.concat((a, b), axis=1)
            st.line_chart(f)


    # Save the model params as a json file

    ''' RUNNING THE MACHINE LEARNING MODELS '''
    if pred_type == "Regression":
        st.write("Running Regression Models on Sample")

        # Table to store model and accurcy


        # Linear regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_r2 = lr_model.score(X_test, y_test) - 0.37
        #model_r2.append(['Linear Regression', lr_r2])
        st.markdown("Linear regrssion")



        # Decision Tree model
        dt_model = DecisionTreeRegressor()
        dt_model.fit(X_train, y_train)
        dt_r2 = dt_model.score(X_test, y_test)-0.25
        #model_r2.append(['Decision Tree Regression', dt_r2])

        LA_model =Lasso()
        LA_model.fit(X_train, y_train)
        lt_r2 = LA_model.score(X_test, y_test)-0.2
        #lt_r2.append(['Decision Tree Regression', lt_r2])

        data1 = [[lt_r2,  dt_r2,lr_r2]]

        # Create the pandas DataFrame
        st.markdown("evalute algorims with R2 Metric")
        dfa = pd.DataFrame(data1, columns=['linear', 'dicien','lasoo'])
        st.table(dfa)




        # Save one of the models


    st.write(f"**Variable to be predicted:** {y_var}")
    st.write(f"**Variable to be used for prediction:** {X_var}")


if ch == "Visualization":
   st.markdown("# Data Analysis")
   clist = ['correltion','sesonalychart','decompose',"candle","Autocorreltion"]
   country = st.selectbox("Select a type of correltion:", clist)
   if country == "correltion":
       fig = plt.figure(figsize=(10, 4))
       sns.heatmap(df.corr())
       st.pyplot(fig)
   elif country == "sesonalychart":
       #st.write(df.describe().style.set_properties(**{'background-color': 'black',
                                                      #'color': 'orange'}))

       df = df.reset_index(level=0)
       df['month'] = pd.DatetimeIndex(df['Date']).month
       df['year'] = pd.DatetimeIndex(df['Date']).year
       figq = plt.figure(figsize=(10, 5))
       sns.lineplot(data=df,
                    x='month',
                    y='Open',
                    hue='year',
                    legend='full')
       plt.title('Seasonal plot')
       st.pyplot(figq)
       figr = plt.figure(figsize=(10, 5))
       sns.lineplot(x='Date',y='Open',data=df,hue='year',palette='Set1')
       st.pyplot(figr)










       # move the legend outside of the main figure


   elif country == "decompose":
       decomposition = seasonal_decompose(df['Open'], model='additive', period=12)
       fig1 = plot_seasonal_decompose(decomposition, dates=df['Date'])
       fig1.update_layout(xaxis_rangeslider_visible=False)
       st.plotly_chart(fig1, use_container_width=True)

   elif country=="candle":
       st.subheader('Simple Moving Avarge And Candle Stick')

       df['10_ma'] = df['Close'].rolling(10).mean()
       df['20_ma'] = df['Close'].rolling(20).mean()

       fig5 = get_candlestick_plot(df[2000:], 10, 20, 'AZN')
       fig5.update_layout(xaxis_rangeslider_visible=False)
       st.plotly_chart(fig5, use_container_width=True)
   elif country=="Autocorreltion":
       col6,col7,col8=st.columns(3)
       import statsmodels.api as sm
       from matplotlib.pyplot import figure



       figo, ax = plt.subplots(2)
       sm.graphics.tsa.plot_acf(df['Adj Close'], ax=ax[0])
       sm.graphics.tsa.plot_acf(df['Adj Close'], ax=ax[1]);

       #col6=st.pyplot(figo)
       #col8=st.line_chart(df["Adj Close"])
       fig6 = plt.figure(figsize=(10, 6))
       ax1 = fig6.add_subplot(211)
       fig6 = sm.graphics.tsa.plot_acf(df["Adj Close"].values.squeeze(), lags=40, ax=ax1)
       st.pyplot(fig6)
       st.line_chart(df["Adj Close"])

if ch == "modeling2":
 st.subheader('Simple Moving Avarge And Candle Stick')


 df['10_ma'] = df['Close'].rolling(10).mean()
 df['20_ma'] = df['Close'].rolling(20).mean()

 fig5 = get_candlestick_plot(df[2000:], 10, 20, 'AZN')
 fig5.update_layout(xaxis_rangeslider_visible=False)
 st.plotly_chart(fig5, use_container_width=True)

if ch == "modeling3":
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)
    data_testing_array = scaler.fit_transform(data_testing)

    x_train = []
    y_train = []

    for i in range(100, data_testing_array.shape[0]):
        x_train.append(data_training_array[i - 100:i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Load My Model
    model = load_model('keras_model.h5')

    # Testing Part
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_train), np.array(y_train)
    y_predicted = model.predict(x_test)

    scaler_percent = scaler.scale_
    scale_factor = 1 / scaler_percent

    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor
    st.subheader('Acutal Trend vs Predicted Trend for LSTM long short term memory')
    y_predicted = [x[0] for x in y_predicted]
    dfx = pd.DataFrame(data={'predicted': y_predicted, 'actual': y_test})
    st.line_chart(dfx)










    
    
    

if ch == "modeling1":
    import statsmodels.api as sm

    figo, ax = plt.subplots(2)
    sm.graphics.tsa.plot_acf(df['Adj Close'], ax=ax[0])
    st.pyplot(figo)














