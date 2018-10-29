# GMM

Starting from a multivariate time series, the objective is to identify a finite number of Gaussian mixtures from which the observations are realized, each one corresponding to a "regime" or market state. Here the parameters of the mixtures are obtained using sklearn's implementation of the Expectation Maximization algorithm (details here: http://scikit-learn.org/stable/modules/mixture.html). 

### Application to Finance: ### 

Armed with an idea of the PDFs behind a time series, one can make better informed decisions about things like hedging (ex: moving from a low volatility regime to a high volatility regime would all else equal favour a long vega/convexity position), as well as portfolio allocation between risky (high vol) and safe (low vol) assets.

### Data: ###
Here I use daily observations of the USD/JPY exchange rate, the 10Y constant maturity JGB yield, and the Nikkei 225 Index.
The timespan is 1971 to 2017.
