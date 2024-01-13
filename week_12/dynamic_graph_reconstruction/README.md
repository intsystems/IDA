### Title: 
Pruning dynamic graph via sampling, Reconstruction of time series via sampling dynamic graph , Reconstruction of time series with pruned dynamic graph, Comparison of two pruning dynamic graph methods via sampling and low-rank decomposition
### Problem:
Let it is given time series. One has to make dynamic graph building pairwise distance matrix, compare methods of pruning: low-rank decomposition to sparce matirx, sampling (distance should be scaled from 0 to 1), and reconstruct obtained dynamic graph to initial time series according to MSE
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
Add sparsity prior to sampling process
### Novelty:
New method pruning dense matrices using Bayesian approach
