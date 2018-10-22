import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import numpy as np
import sklearn.mixture as mix
from matplotlib.dates import YearLocator, MonthLocator
import warnings
from pylab import rcParams
from matplotlib.pyplot import cm
rcParams['figure.figsize'] = [10, 5]

sns.set() 
warnings.filterwarnings("ignore") 

paths = ["/Users/julienraffaud/Desktop/Data/DEXJPUS.csv",
         "/Users/julienraffaud/Desktop/Data/DGS10.csv",
         "/Users/julienraffaud/Desktop/Data/NIKKEI225.csv"]

# Importing, collating and formatting our three time series:
# USD/JPY, 10Y constant maturity JGB yield & Nikkei index
dataframes = []
for file in paths:
    df = pd.read_csv(file)
    df["DATE"] = df["DATE"].astype(datetime) 
    df = df[df[df.columns[1]]!= "."]
    df[df.columns[1]] = df[df.columns[1]].astype(float)
    #df[df.columns[1]] = np.log(df[df.columns[1]]) - np.log(df[df.columns[1]].shift(1))
    df = df.set_index("DATE")
    dataframes.append(df)

# Formatting the final dataframe of the time series
time_series = pd.concat(dataframes,axis=1,sort='False').dropna() 
time_series.columns = ["USD/JPY","JGB","NIKKEI"]
time_series.index = pd.to_datetime(time_series.index)

# Making a dataframe of the time series' log-returns
log_series = time_series.copy(deep="True")
log_series[log_series.columns] = np.log(log_series[log_series.columns]) - np.log(log_series[log_series.columns].shift(1))
log_series = log_series.dropna()

# Fitting the hidden markov model to the data.
# Here I set the number of components to 2, in order
# to account for normal and high volatility regimes.
X = log_series[log_series.columns].values
model = mix.GaussianMixture(n_components=4,
                            covariance_type='full',
                            n_init=100,
                            random_state = 711).fit(X)
hidden_states = model.predict(X)

# Displaying the parameters of the 2 fitted states.
# Note first state is our normal volatility regime whilst the second is
# the stressed markets regime, with distinctly higher volatility 
# and negative average log-return.
print("Mean and variance of each hidden state:")
for i in range(model.n_components):
    print("{0}th hidden state:".format(i+1))
    df = pd.DataFrame({"Mean":model.means_[i]*252,
                       "Variance":np.diag(model.covariances_[i])}).T
    df.columns = time_series.columns
    print(df)

# Adding the state column to the time series dataframes
states = pd.DataFrame(hidden_states,index=log_series.index)
log_series = pd.concat([log_series,states],axis=1,sort='False').dropna()
log_series.rename(columns={0:'state'}, inplace=True)
time_series = pd.concat([time_series,states],axis=1,sort='False').dropna()
time_series.rename(columns={0:'state'}, inplace=True)

# Plotting the empirical distribution of log-returns 
# associated with each state.
for asset in time_series.columns[:-1]:
    for i in range(model.n_components):
        mask = log_series.state==i
        sns.distplot(log_series[mask][asset],hist=True,kde=True)
    plt.legend(["state "+str(i) for i in range(model.n_components)])
    plt.xlabel(asset+" log returns")
    plt.show()
    
# Plotting the time series shaded by state.
color=cm.rainbow(np.linspace(0,1,model.n_components))
for asset in time_series.columns[:-3]:
    for i in range(model.n_components):
        mask = time_series.state==i
        plt.plot(time_series[mask].index,
                   time_series[mask][asset],
                   c=color[i],
                   marker=".")
        plt.title(asset+"- State "+str(i+1))
        plt.show()

