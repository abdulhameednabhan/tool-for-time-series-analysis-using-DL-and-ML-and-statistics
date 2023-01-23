import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
import seaborn as sns
import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.patches as mpatches
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from pandas_datareader import data
from datetime import datetime, timedelta
import plotly
import pandas as pd
import  joblib
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import streamlit as st
from pandas_datareader import data
import datetime




import plotly.graph_objects as go
df = pd.read_csv("azn1.csv")
correltion = pd.read_csv("azn1.csv")
coverince=pd.read_csv("azn1.csv")


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
st.header('time series\stock data')



with st.sidebar:
    ch= option_menu("Time Series Analysis ", ["browsing data", "upload real", "modeling", "modeling2", "about"],
                       icons=['house', 'camera fill', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "##00172B"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "###00172B"},
        "nav-link-selected": {"background-color": "#02ab21"},
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

if ch == "upload real":
    stock_name = st.text_input("Enter the stock name: \n", 'AAPL')
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








    """

    # Visualizations
    
    st.subheader('Closing Price vs Time Chart')
    st.line_chart(df2.Close)

    # Visualizations
    st.subheader('Closing Price vs Time Chart with 100MA')
    mma100 = df2.Close.rolling(100).mean()
    df.insert(6, "mma100", mma100, True)
    st.line_chart(df2[['Close', 'mma100']])
    """
if ch == "modeling":
   data = pd.read_csv('azn1.csv')
   model_r2 = []

   # Create the model parameters dictionary
   params = {}

   # Use two column technique
   col1, col2 = st.columns(2)

   # Design column 1
   y_var = col1.radio("Select the variable to be predicted (y)", options=data.columns)

   # Design column 2
   X_var = col2.multiselect("Select the variables to be used for prediction (X)", options=data.columns)

   # Check if len of x is not zero
   if len(X_var) == 0:
       st.error("You have to put in some X variable and it cannot be left empty.")

   # Check if y not in X
   if y_var in X_var:
       st.error("Warning! Y variable cannot be present in your X-variable.")

   # Option to select predition type
   pred_type = st.radio("Select the type of process you want to run.",
                        options=["Regression", "Classification"],
                        help="Write about reg and classification")

   # Add to model parameters
   params = {
       'X': X_var,
       'y': y_var,
       'pred_type': pred_type,
   }

   # if st.button("Run Models"):

   st.write(f"**Variable to be predicted:** {y_var}")
   st.write(f"**Variable to be used for prediction:** {X_var}")
   X = data[X_var]
   y = data[y_var]

   st.markdown("#### Train Test Splitting")
   size = st.slider("Percentage of value division",
                    min_value=0.1,
                    max_value=0.9,
                    step=0.1,
                    value=0.8,
                    help="This is the value which will be used to divide the data for training and testing. Default = 80%")

   X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
   st.write("Number of training samples:", X_train.shape[0],X_train.shape[1])
   st.write("Number of testing samples:", X_test.shape[0],X_test.shape[1],y_test.shape[0])
   lr_model = LinearRegression()
   lr_model.fit(X_train, y_train)
   lr_r2 = lr_model.score(X_test, y_test)
   lr_model = LinearRegression()
   lr_model.fit(X_train, y_train)
   lr_r2 = lr_model.score(X_test, y_test)
   model_r2.append(['Linear Regression', lr_r2])

   # Decision Tree model
   dt_model = DecisionTreeRegressor()
   dt_model.fit(X_train, y_train)
   dt_r2 = dt_model.score(X_test, y_test)
   model_r2.append(['Decision Tree Regression', dt_r2])

   # Save one of the models
   if dt_r2 > lr_r2:
       # save decision tree
       joblib.dump(dt_model, 'data/metadata/model_reg.sav')
   else:
       joblib.dump(lr_model, 'data/metadata/model_reg.sav')

   # Make a dataframe of results
   results = pd.DataFrame(model_r2, columns=['Models', 'R2 Score']).sort_values(by='R2 Score', ascending=False)
   st.dataframe(results)






if ch == "modeling2":
  st.markdown("# Data Analysis")
  clist = ['correltion','coverince']
  country = st.selectbox("Select a type of correltion:", clist)
  if country == "correltion":
      fig = plt.figure(figsize=(10, 4))
      sns.heatmap(df.corr())
      st.pyplot(fig)
  elif country == "coverince":
      def plot_seasonal_decompose(result: DecomposeResult, dates: pd.Series = None,
                                  title: str = "Seasonal Decomposition"):
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
                  height=900, title=f'<b>{title}</b>', margin={'t': 100}, title_x=0.5, showlegend=False
              )
          )


      decomposition = seasonal_decompose(df['Open'], model='additive', period=12)
      fig = plot_seasonal_decompose(decomposition, dates=df['Date'])
      st.plotly_chart(fig, use_container_width=True)

if ch == "about":

   def plot_seasonal_decompose(result: DecomposeResult, dates: pd.Series = None,
                               title: str = "Seasonal Decomposition"):
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
               height=900, title=f'<b>{title}</b>', margin={'t': 100}, title_x=0.5, showlegend=False
           )
       )





   decomposition = seasonal_decompose(df['Open'],model='multipitive', period=128)
   fig = plot_seasonal_decompose(decomposition, dates=df['Date'])
   st.plotly_chart(fig, use_container_width=True)

   import streamlit as st
   from pandas_datareader import data
   import datetime

   ###########
   # sidebar #
   ###########
   #
   ticker_input = st.selectbox('Select one symbol', ('AAPL',))
   # create default date range
   start = datetime.datetime(2010, 1, 1)
   end = datetime.datetime(2021, 11, 1)
   # ask user for his date
   start_date = st.date_input('Start date', start)
   end_date = st.date_input('End date', end)
   # validate start_date date is smaller then end_date

   st.title('Close, Open, Low, High')
   # get data based on dates
   df = data.DataReader(ticker_input, 'yahoo', start_date, end_date)
   # plot
   st.line_chart(df.loc[:, ["Close", "Open", "Low", "High"]])






#decomposition = seasonal_decompose(df['Open'],model='multipitive', period=128)

#fig=decomposition.plot()




#st.plotly_chart(fig, use_container_width=True)
#seasonal_decompositions = seasonal_decompose(df['Open'], model='multipitive', period=128)
#seasonal_decomposition_fig = seasonal_decompositions.plot()
#seasonal_decomposition_fig = plotly.tools.mpl_to_plotly(seasonal_decomposition_fig)
#seasonal_decomposition_fig.update_layout(width = 1100, height = 500, title = 'Seasonal Decomposition')


#st.plotly_chart(seasonal_decomposition_fig, use_container_width=True)













