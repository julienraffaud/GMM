# GHMM
Analysing regime switches in multivariate time-series using a Gaussian Hidden Markov Model. Here I use USDJPY,
10Y constant maturity JGB yield, and NIKKEI Index time series from 1971 to late 2017.

The objective is to detect when asset prices' log-returns shift from one probability density to another, with
each probability density associated with one "regime", or state. Armed with this knowledge one can make better informed decisions about hedging (ex: moving from a low volatility state to a high volatility state would all else equal favour 
a long vega/gamma position) and portfolio allocation between risky (high volatility) and safe (low vol) assets. 
