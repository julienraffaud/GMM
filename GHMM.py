import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
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
# USD/JPY, 10Y constant maturity JGB yield & Nikkei index.
dataframes = []
for file in paths:
    df = pd.read_csv(file)
    df["DATE"] = df["DATE"].astype(datetime) 
    df = df[df[df.columns[1]]!= "."]
    df[df.columns[1]] = df[df.columns[1]].astype(float)
    #df[df.columns[1]] = np.log(df[df.columns[1]]) - np.log(df[df.columns[1]].shift(1))
    df = df.set_index("DATE")
    dataframes.append(df)

# Formatting the final dataframe of the time series.
time_series = pd.concat(dataframes,axis=1,sort='False').dropna() 
time_series.columns = ["USD/JPY","JGB","NIKKEI"]
time_series.index = pd.to_datetime(time_series.index)

# Making a dataframe of the time series' log-returns.
log_series = time_series.copy(deep="True")
log_series[log_series.columns] = np.log(log_series[log_series.columns]) - np.log(log_series[log_series.columns].shift(1))
log_series = log_series.dropna()

# To determine the appropriate covariance type & number of components our Gaussian Mixture Model
# must feature we can use the information-theoretic criteria and rank the accuracy of different fits 
# to our data:
lowest_bic = np.infty
bic = []
n_components_range = range(1, 10)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        model = mix.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        model.fit(X)
        bic.append(model.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_ = model

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_
bars = []

# Plotting the BIC scores.
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

# Fitting the Gaussian Mixture Model to the data.
# Here I set the number of components to 4 and covariance type to full,
# because the marginal BIC score improvement for added components was low
# and I want to avoid overcomplicating the model.
X = log_series[log_series.columns].values
model = mix.GaussianMixture(n_components=2,
                            covariance_type='full',
                            n_init=100,
                            random_state = 711).fit(X)

#Predicting the hidden states.
hidden_states = model.predict(X)

# Displaying the parameters of the fitted mixtures.
print("Mean and variance of each hidden state:")
for i in range(model.n_components):
    print("{0}th hidden state:".format(i+1))
    df = pd.DataFrame({"Mean":model.means_[i]*252,
                       "Variance":np.diag(model.covariances_[i])}).T
    df.columns = time_series.columns
    print(df)

# Adding the predicted hidden states to the time series dataframes.
states = pd.DataFrame(hidden_states,index=log_series.index)
log_series = pd.concat([log_series,states],axis=1,sort='False').dropna()
log_series.rename(columns={0:'state'}, inplace=True)
time_series = pd.concat([time_series,states],axis=1,sort='False').dropna()
time_series.rename(columns={0:'state'}, inplace=True)

# Plotting the empirical distribution of log-returns
# associated with each hidden state.
for asset in time_series.columns[:-1]:
    for i in range(model.n_components):
        mask = log_series.state==i
        sns.distplot(log_series[mask][asset],hist=True,kde=True)
    plt.legend(["state "+str(i) for i in range(model.n_components)])
    plt.xlabel(asset+" log returns")
    plt.show()
    
# Plotting the time series using a scatterplot, with observations coloured by hidden state.
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
