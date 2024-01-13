### Title: 
Pruning dynamic graph via sampling, Reconstruction of time series via sampling dynamic graph, Reconstruction of time series with pruned dynamic graph, Comparison of two pruning dynamic graph methods via sampling and low-rank decomposition
### Problem:
Let it is given time series. One has to:
1. using neural network _f_ build sparse stochastic matrix, where vecotrs would be discrete pobabilty ditsribution of edges between two nodes (will look like pairwise distance matrix);
2. get top p graph edges from these distribution (probably it could be skiped, because obtained stochastic matrix could already be used as adjenecy matrix);
3. reconstruct obtained dynamic graph to initial time series according to MSE
### Data: 
Motor Imagery datasets  from MOABB framework: 
1. [AlexMI](https://neurotechx.github.io/moabb/generated/moabb.datasets.AlexMI.html#moabb.datasets.AlexMI) - for simple experiments
2. [Lee2019MI](https://neurotechx.github.io/moabb/generated/moabb.datasets.Lee2019_MI.html#moabb.datasets.Lee2019_MI) - for actual banchmark
### References:
1. Masters Thesis Varenik 2022 ([link](http://www.machinelearning.ru/wiki/images/b/b2/Varenik2022master_thesis.pdf))
2. Discrete Graph Structure Learning for Forecasting Multiple Time Series 2021 ([link](https://arxiv.org/abs/2101.06861))
### Base solution:
1. implementation of methods without pruning;
2. implementation from Varenik 2022
### Proposed solution:
The eaysiest way to make adjenecy matrices sparse is to make stochastic matrices sparse too. To do it we add regularizer for first neural network _f_, accordingly to a prior distirbution which mode is close to zero. 
### Novelty:
This method gives ability to sample graphs from time series that are already sparsed by the additional regularizers 
