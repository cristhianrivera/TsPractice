# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:13:28 2018

@author: a688291
"""



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd # generating random numbers
import datetime # manipulating date formats
# Viz
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots


# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.tsa as smar
import statsmodels.api as sm
import scipy.stats as scs


# settings
import warnings
import os
warnings.filterwarnings("ignore")

os.chdir("C:/Users/a688291/Documents/EDA_CRJR/Practice/Python/TsPractice")
os.getcwd()

# Import all of them 
sales=pd.read_csv("../input/sales_train_v2.csv")
item_cat=pd.read_csv("../input/item_categories.csv")
item=pd.read_csv("../input/items.csv")
sub=pd.read_csv("../input/sample_submission.csv")
shops=pd.read_csv("../input/shops.csv")
test=pd.read_csv("../input/test.csv")



sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
# check
sales.info()
item_cat.info()

# Aggregate to monthly level the required metrics

monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[
    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})




# number of items per cat 
x=item.groupby(['item_category_id']).count()
x=x.sort_values(by='item_id',ascending=False)
x=x.iloc[0:10].reset_index()
x
# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)
plt.title("Items per Category")
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()


# number of items per store per month
monthly_items_store=sales.groupby(["date_block_num","shop_id"])["item_cnt_day"].agg({"item_cnt_day":"sum"})
monthly_items_store.head(40)

##First example of time series for the total amount of items sold by the company
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total Sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts);


plt.figure(figsize=(16,6))
plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');
plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');
plt.legend();



import statsmodels.api as sm
# multiplicative
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")
fig = res.plot()


# Stationarity tests
def test_stationarity(timeseries):
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(ts)



ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=pd.Series.diff(ts)
plt.plot(new_ts)
plt.plot()

plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=pd.Series.diff(ts,12)       # assuming the seasonality is 12 months long
plt.plot(new_ts)
plt.plot()


# now testing the stationarity again after de-seasonality
test_stationarity(pd.Series.dropna(new_ts)) #drop the nans
#p-value                         0.016269 ready to forecast




def tsplot(y, lags=None, figsize=(10, 8), style='bmh',title=''):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 


#AR(1) simulated process
np.random.seed(1)
n_samples = int(1000)
a = 0.6
x = w = np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t] = a*x[t-1] + w[t]
limit=12    
_ = tsplot(x, lags=limit,title="AR(1)process")

#MA(2) simulated process
n = int(1000)
alphas = np.array([0.])
betas = np.array([0.6, 0.4])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma3 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
_ = tsplot(ma3, lags=12,title="MA(2) process")




###--------------------------------------------------------------------

###ARIMA with prophet
from fbprophet import Prophet
#prophet reqiures a pandas df at the below config 
# ( date column named as DS and the value column as Y)

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
ts=ts.reset_index()
ts.head()
ts.columns=['ds','y']
model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
model.fit(ts) #fit the model with your dataframe

future = model.make_future_dataframe(periods = 5, freq = 'MS')  
# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

model.plot(forecast)







# pick best order by aic 
# smallest aic value wins
best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(pd.Series.dropna(new_ts).values, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue
print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))



plt.plot(new_ts)
plt.plot( best_mdl.predict())

nts = pd.concat([pd.Series.dropna(new_ts), pd.Series(best_mdl.predict())], ignore_index=True)
plt.plot(nts)


new_ts
pd.Series(new_ts)




##------------------------------------------------------
##ARIMA Models with statsmodels.tsa

def PlotSeries(ts, lags=12):
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)
    y = pd.Series.dropna(ts)
    plt.figure(figsize=(10,8))
    plt.subplot(311)
    layout = (2,2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    y.plot(ax=ts_ax)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=1, zero=False)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=1, zero=False)
    plt.tight_layout()

PlotSeries(ts)

## Series transformation
plt.figure(figsize=(12,7))
ts = sales.groupby(["date"])["item_cnt_day"].sum()
ts.plot()

plt.figure(figsize=(12,7))
log_ts = np.log(ts)
log_ts.plot()

plt.figure(figsize=(12,7))
diff_ts=pd.Series.diff(log_ts, periods = 7)
diff_ts.plot()

plt.figure(figsize=(12,7))
forw_ts = pd.Series(np.r_[log_ts.iloc[0],diff_ts.iloc[7:]].cumsum())
forw_ts.plot()

##
plt.figure(figsize=(12,7))
diff_ts_7 = np.log((pd.Series.dropna(diff_ts))+1)
diff_ts_7.plot()
diff_ts_7_1=pd.Series.diff(ts, periods =1)
PlotSeries(diff_ts_7_1)

##


TSmodel = smar.arima_model.ARIMA(diff_ts_7,order=(0,1,1))
TSmodelFit = TSmodel.fit(disp=0)
print(TSmodelFit.summary())
# plot residual errors
residuals = pd.DataFrame(TSmodelFit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())
PlotSeries(residuals)


np.r_[ts.iloc[0:7],diff_ts.iloc[7:]].cumsum()



df_ts = df_ts.assign(diff_ts= diff_ts.values)
df_ts = df_ts.assign(for_ts = df_forw_ts.values)


pd.DataFrame.append


diff_ts.plot()

forw_ts = np.r_[ts.iloc[0],diff_ts.iloc[1:]].cumsum()
pd.concat([pd.Series(ts.iloc[0]),forw_ts],ignore_index=True)
forw_ts = pd.Series(forw_ts)
forw_ts.plot()






x, x_diff = df['A'].iloc[0], df['B'].iloc[1:]
df['C'] = np.r_[x, x_diff].cumsum()

new_ts.plot()
new_ts=pd.Series.dropna(pd.Series.diff(new_ts, periods = -7))
new_ts.plot()


new_ts=pd.Series.dropna(pd.Series.diff(new_ts, periods = 52))
PlotSeries(new_ts)

ts = new_ts

TSmodel = smar.arima_model.ARIMA(ts,order=(0,1,1))
TSmodelFit = TSmodel.fit(disp=0)
print(TSmodelFit.summary())
# plot residual errors
residuals = pd.DataFrame(TSmodelFit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())
PlotSeries(residuals)














