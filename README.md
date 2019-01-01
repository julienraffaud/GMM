# GMM

The objective is to detect regimes in arbitrarily large multivariate time series by "greedily" partitioning the time series into segments, with each segment's data consisting of independent samples from a multivariate Gaussian distribution. 

### Data: ###
Here I use daily log-returns of the USD/JPY exchange rate, the 10Y constant maturity JGB yield, and the Nikkei 225 Index.
The timespan is 1971 to 2017.

### Academic Reference:

Hallac D., Nystrup P., Boyd S.
2018. Greedy Gaussian Segmentation of Multivariate Time Series. (Arxiv: https://arxiv.org/pdf/1610.07435.pdf).
