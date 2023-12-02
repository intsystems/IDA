# Intelligent data analysis

Classes: every Saturday, 11:15

[Zoom link](m1p.org/go_zoom)

**Topic of this year:** Brain-computer interface.

Brain-computer interface with Functional data analysis Human behavioral
analysis and forecasting require models that have to predict the target
variables of complex structures. We develop PLS and CCA (Projection to
latent space and Canonic correlation analysis) methods towards the
Multiview with continuous-time data representation.

## Generative time series decoding models
--------------------------------------

1.  **The goal** is to create a generative state-space model for BCI
2.  **The impact** is to boost the behavioral classification quality by
    decision-rejecting
3.  **The principle** if a generated pattern does not belong to one of
    the expected patterns (one-class classification), we reject the
    decision
4.  **The plan**
    1.  create the simplest generative model for selected data
    2.  apply SSM (state-space model) principles to make CCA
        (canonic-correlation analysis)
    3.  introduce the classification model and decision-rejecting criterion
    4.  compare quality

### References

1.  Direct Discriminative Decoder Models for Analysis of
    High-Dimensional Dynamical Neural Data by M.R. Rezaei et al. 2022
    \[DOI <https://doi.org/10.1162/neco_a_01491>\]
2.  Deep Direct Discriminative Decoders for High-dimensional Time-series
    Data Analysis by M.R. Rezaei 2023 (NTDB)
3.  Decoding Hidden Cognitive States From Behavior and Physiology Using
    a Bayesian Approach by Ali Yousefi et al.
    [DOI](https://doi.org/10.1162/neco_a_01196)
4.  Bayesian Decoder Models with a Discriminative Observation Process by
    M.R. Rezaei et al. 2020
    [DOI](https://doi.org/10.1101/2020.07.11.198564)
5.  Deep Discriminative Direct Decoders for High-dimensional Time-series
    Analysis by M.R. Rezaei 2020
    [ArXiv](https://arxiv.org/abs/2205.10947)
6.  Real-Time Point Process Filter for Multidimensional Decoding
    Problems Using Mixture Models by M.R. Rezaei 2020
    [DOI](https://doi.org/10.1016/j.jneumeth.2020.109006)
7.  [Basic code
    D4](https://github.com/vadim-vic/Deep_Direct_Discriminative_Decoder-D4-)
8.  Variational auto-encoded deep Gaussian processes by Z. Dai et al.
    2021 [ArXiv](https://arxiv.org/abs/1511.06455)
9.  Parametric Gaussian process regressors by M. Jankowiak et al. 2020
    [ArXiv](https://arxiv.org/abs/1910.07123)
10. A Tutorial on Gaussian Processes by Z. Ghahramani 2010
    [PDF](http://learning.eng.cam.ac.uk/zoubin)
11. An Intuitive Tutorial to Gaussian Processes Regression by J. Wang
    2021 [ArXiv](https://arxiv.org/abs/2009.10862)

## Riemannian Geometry and Graph Laplacian metric models
-----------------------------------------------------

1.  **The goal** is to create a metric **behavioral** forecasting model
    for BCI
2.  **The impact** is to construct time-embedding metric space so that
    it is compatible with the generative models
3.  **The principle** a dynamic system changes its state consequently,
    so we construct a metric state space that could be decomposed with
    one of the diffusion models
4.  **The plan**
    1.  select a metric model with continuous time
    2.  use Riemannian geometry and Graph-Laplacian approaches
    3.  make diffusion decomposition
    4.  boost decoding models with metric space

### References

1.  Classification of covariance matrices using a Riemannian-based
    kernel for BCI applications by A. Barachant et al. 2013
    (Neurocomputing)
2.  Multi-class Brain-Computer Interface Classification by Riemannian
    Geometry by A. Barachant et al.
3.  Riemannian Geometry for EEG-based Brain-Computer Interfaces by M.
    Congedo et al.
4.  Online SSVEP-based BCI using Riemannian geometry by E. K. Kalunga
    2016 [DOI](http://dx.doi.org/10.1016/j.neucom.2016.01.007)
5.  A Plug&Play P300 BCI Using Information Geometry by A. Barachant 2014
    [ArXiv](https://arxiv.org/abs/1409.0107)
6.  Longitudinal predictive modeling of tau progression along the
    structural connectome by J.Dutta et al. 2021
    [DOI](https://doi.org/10.1016/j.neuroimage.2021.118126)
7.  Grand: Graph neural diffusion by M.M. Bronstein et al. ICML, 2021.
8.  (inspiring) The inverse problem in electroencephalography using the
    bidomain model of electrical activity by A.L. Rincon and S. Shimoda,
    2016 [DOI](http://dx.doi.org/10.1016/j.jneumeth.2016.09.011)
9.  (inverse) High-Resolution EEG Source Reconstruction with Boundary
    Element Fast Multipole Method, N. Makaroff et al. 2022
    [DOI](https://doi.org/10.1101/2022.10.30.514418)

## Data
----

Any data that has

1.  timeline with a behavioral pattern, synchronous both for source and
    target data
2.  source time series with
    -   probabilistic assumptions for diffusion probabilistic models
3.  target time series with
    -   behavioral pattern to make a decision

To select from

-   [List of datasets and tools](BCI "wikilink")
-   Scientific lessons from a catalog of 6674 brain recordings by A.D.J. Makin

## The problem
-----------

1.  To make a classification decision or to reject it
2.  To forecast a system behavior (system state) and generate variants
3.  The rejection criterion is a mismatch observation from generated
    scenarios

Assumptions
-----------

1.  Short time series (relatively, hundreds or thousands of samples)
2.  Time series have big variances and systematic errors
3.  Time series could be significantly correlated
4.  Time series have origins
    1.  exogenous, no one can control
    2.  control signals
    3.  and decisions
    4.  behavioral
5.  Timeline has structure
    1.  periods (seasonal or quasi)
    2.  events (forced or selected)
    3.  including decision timestamps

### Examples
--------

1.  Any sports game (exogenous, behavior to model, decisions)
2.  Brain-computer interfaces
3.  Risk management, stock market   

### The challenge is to put
-----------------------

-   generative models
-   of time series graph Laplacian
-   into seq2seq framework
-   to make quality decisions.

## Schedule 2023


| Date       |     | N   | Progress                             | To discuss                                                    | Result                                          |
|------------|-----|-----|--------------------------------------|---------------------------------------------------------------|-------------------------------------------------|
| September  | 16  | 1   | Introduction and planning            | Topics for the next week                                      | Subscribed to the schedule                      |
|            | 23  | 2   | Possible models                      | Introductions to models                                       | Slides, references, questions to reason         |
|            | 30  | 3   | Model reasoning and selection        | Discussion of the questions, refined presentations of models  | Role of models in the system                    |
| October    | 14  | 4   | System description                   | Variants of the system, code sources                          | List of code references, toolbox(es) selection  |
|            | 21  | 5   | Architecture planning                | Variants of the pipelines                                     | Collection of pipelines, executable             |
|            | 28  | 6   | Data collection                      | Discussion of the applications                                | Links to the data and downloaders               |
| November   | 11  | 7   | Minimum value system                 | Discussion of the tutorials                                   | Tutorial plan                                   |
|            | 18  | 8   | Examples and documents               | Tutorial presentation                                         | List of new models and challenges               |
|            | 25  | 9   | New directions                       | Model presentation (week 2)                                   | List of challenges                              |
| December   | 2   | 10  | Analysis, testing, and applications  | Decomposition of the system                                   | List of possible projects                       |
|            | 9   | 11  | Project proposals                    | Project presentations                                         | Project descriptions                            |
|            | 16  | 12  | Project description review           | Discussion of the messages                                    | Published descriptions                          |

### Week 1. Motivation and projects
* [Course motivation and projects  to select from](https://github.com/intsystems/IDA/blob/main-2023/week_1/Goals_and_topics.odt)

### Week 2. Topics to discuss

1.  CCA generative models
2.  Continous models for graph Laplacian
3.  D4 and variants
4.  Riemmanian model with time
5.  SSM generative models
6.  Generative models with time graph convolutions
7.  (risky) RL for time series scenario generation
8.  (math) CCA Error functions and (tech) Autograd for seq2seq
    generative pipelines

### Week 3. Topics to discuss
Reason variations of models with a model description and a table of pros and cons. 
*  CCA generative models: compare versions of generation techniques (variational, flow, and diffusion)
*  Continous models for graph Laplacian: compare with Riemannian approach and CGN
*  D4 and variants: compare Kalman filter, S4, HiPPO in discrete and continous time

#### Examples of editorial-accepted comparisons:
Just see Table 1 and its surrounding explanatory text only.
1. Generation of simple structured Information Retrieval functions by genetic algorithm without stagnation by  A. Kulunchakov and V. Strijov. [DOI](https://doi.org/10.1016/j.eswa.2017.05.019), [PDF](http://strijov.com/papers/Kulunchakov2014RankingBySimpleFun.pdf)
2. Multi-way feature selection for ECoG-based Brain-Computer Interface by A. Motrenko, V. Strijov. [DOI](https://doi.org/10.1016/j.eswa.2018.06.054), [PDF](http://strijov.com/papers/MotrenkoStrijov2017ECoG_HL_2.pdf)
3. Quadratic programming feature selection for multicorrelated signal decoding with partial least squares by R. Isachenko, V. Strijov. [DOI](https://doi.org/10.1016/j.eswa.2022.117967), [PDF](http://strijov.com/papers/isachenko2022qpfs_decoding.pdf)

#### Expected results:
1. Short review of the listed alternative models (and the other alternatives you may find)
2. Comparative analysis; see Table 1 in the papers above for your inspiration

(Note. Please think of possible model structure selection and dimensionality reduction, especially in the target space)

### Week 4. Topics to discuss
List of toolboxes and its readiness to use in pipelines

Types of toolboxes to fill in
1. CCA: Canonical correlation analysis (and generative)
2. GEN: Generative models: 1) Normalizing flows and Autoregressive models, 2) Probabilistic diffusion models and Variational autoencoders
3. GBM: Graph and dynamic barycenter models
4. SSM: State-space models 1) discrete-time, 2) continous-time
5. CCM: Cross-convergence mapping
6. GMS: Generative Bayesian model selection
7. USE: Useful seq2seq models and transformers

| Code | Toolbox    | Doc, Demo | Pipeline | Comment on usability |
|------|------------|-----------|----------|----------------------|
| CCM  | [pyRiemann: Biosignals classification with Riemannian geometry](https://pyriemann.readthedocs.io/en/latest/)       | Y, Y          | sk-learn         | Quick start with useful extensive demos and data ready to go.                     |
| GEN  | [pythae](https://github.com/clementchadebec/benchmark_VAE)                                                   | many tutorials in the original repo                                  | PyTorch     | Plug and play library. Potential applications to our research projects are questionable.                                              |
| GEN  | [DeepCCA](https://github.com/Michaelvll/DeepCCA)                                                             | No docs                                                              | PyTorch     | A straightforward implementation that takes 50 lines of code. Potential applications are promising.                                   |
| SSM     |  [State Spaces](https://github.com/HazyResearch/state-spaces), [D4](https://github.com/vadim-vic/Deep_Direct_Discriminative_Decoder-D4-), [Kalman Filter](https://github.com/strongio/torch-kalman)         | look at README.md page of each of the repo's         | Pytorch-Lightning                     | S4 is tested on real data as a diffusion model module. The operation of the calculation using Cauchy kernels has been tested. D4 and Kalman Filter were tested using basic tests from the demo.
| GBM  | [node2coords](https://github.com/ersisimou/node2coords/tree/master)                                          |                                                                      | PyTorch     | GitHub of the article [node2coords: Graph Representation Learning with Wasserstein Barycenters](https://arxiv.org/pdf/2007.16056.pdf) |
| GBM  | [pytorch-geometric](https://github.com/pyg-team/pytorch_geometric)                                           | [Doc](https://pytorch-geometric.readthedocs.io/en/latest/index.html) | PyTorch     | PyG (PyTorch Geometric) is a library built to easily write and train GNNs                                                             |
| GBM  | [Graph Convolutional Networks in PyTorch](https://github.com/tkipf/pygcn)                                    |                                                                      | PyTorch     | Good example of GCN implementation using PyTorch                                                                                      |
| GBM  | [Graph Convolutional Networks](https://github.com/tkipf/gcn)                                                 |                                                                      | TensorFlow  | Good example of GCN implementation using TensorFlow                                                                                   |
| GBM  | [GNN using PDE](https://github.com/kaist-silab/awesome-graph-pde)                                            |                                                                      | PyTorch/PyG | A collection of resources about partial differential equations, deep learning, graph neural networks, dynamic system simulation       |
| GBM  | [DIFFormer](https://github.com/qitianwu/DIFFormer/tree/main)                                                 |                                                                      | PyTorch/PyG | GitHub of the article [DIFFormer: Diffusion-based (Graph) Transformers](https://arxiv.org/pdf/2301.09474.pdf) 

### Week 5. Topics to discuss
1.	Graph continuously changes its structure. One has to forecast its structure given the past history of changes.
2.	CNN tunes its parameters according to the categorial target. One needs to tune its parameters according to the source data manifold )or a manifold of some other data). The convolution goes over time and 2/3 dim space.
3.	There given a set of time series to reconstruct. To reconstruct these time series (given the significant covariance hypothesis), one has to find distances between time series (points on their phase trajectories), construct SPD(t) distance matrix, prune graph (reducing dimensionality), and reconstruct time series.

### Week 6. Topics to discuss
Pytorch keeps models in the pipeline without an error function. An example in the first lines of the Lightning example. In the encoder-decoder pipeline, an error common to all is determined. For CCA methods, it is important to include the error in an arbitrary module. The priors (according to Bayesian inference) are terms of the error function. Since in CCA, prior distributions are imposed on the generated samples (and participate in the optimization of the generative model, which operates in the latent space), the error function must appear in the latent space.

We discuss the following questions:

1. Composite pipelines suitable for use in CCA methods. Types of Examples of building pipelines in which .transform (with settings) appears inside .fit , and not before optimization?

2. Optimization schedule in the context of CCA. How is this schedule implemented? Need technical information and code examples.

For the first two points, it is advisable to discuss how to decompose CCA into a pipeline with four errors: 1) the left and right PCA matrices are adjusted to the minimum of the mean square error, 3) the latent space is constructed through the maximum of the covariance (as in the description of the method), 4) the predictive model, superposition 1 )-3) is constructed with an error in accuracy. In the linear case, such a decomposition is clearly redundant. Still, if we are talking about stacks of autoencoders, U-net, then such a decomposition is necessary to obtain a good initial approximation of the parameters.

3. Scoring-based diffusion as applied to CCA.

Please add suitable pieces of code and links to the slides. Let's discuss programming technologies.

### Week 7. Topics to discuss
The paper in the field of 

Collect datasets. The requirements for the links:
1. The dataset matches one of the listed problems:
    2. CCA with a vector or tensor source and target spaces
    3. Behavioural data with the autoregressive state space as the target
    4. Multimodal data with a simple structure
    5. Graph data at least in one space, source, or target
3. It has an explicit structure and description
4. It has a downloader (really welcome) or a code for easy download

Code in the table  
1. Spatial-time series for CCA and multomodal problems 
2. GEN, GEN2 stands for Generative models for CCA
3. GBM, GBM2
4. (BEH behavioral: state space for classification, sequence of classes)

Applications and origins of data
1. Brain-computer, high frequency: EEG, ECoG, iEEG, ..., low frequency: fMRI, eye, hand, tracking, ...
2. Mechanical movements: accelerometer, gyroscope, ...
3. Financial time series: the lowest frequency is daily time-stamps 
4. Industrial time series: electricity consumption, device testing, production monitoring, ...
5. Physics measurements, including weather forecasts, space research, high-energy physics, ...
   
| Code | Data       | Downloader | Comment on usability |
|------|------------|------------|----------------------|
|  GBM |    [OGB](https://ogb.stanford.edu)     |    Yes     | Own library, many datasets for different tasks, works with pytorch-geometric out of the box |
| GBM  | [PyTorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#module-torch_geometric_temporal.dataset.chickenpox) |  Yes    |            Fork of pytorch-geometric, works with temporal graphs, has a lot of implemented dataset classes, **NB!** too few commits since 2022 |
|  GEN    |    [KU](http://gigadb.org/dataset/view/id/100542/File_page/7)        |    Yes      |  Extremely trivial to download and make spatial covariance matrices as was done in [Score-based EEG generation](https://arxiv.org/pdf/2302.11410.pdf)      |
| STS  | [EEGEyeNet](https://paperswithcode.com/paper/eegeyenet-a-simultaneous)           | [Yes](https://osf.io/ktv7m/)           |       Has a [benchmark](https://github.com/ardkastrati/eegeyenet)  for evaluating gaze prediction consisting of three tasks.              |
| STS  | [MotionSense](https://paperswithcode.com/dataset/motionsense)           | [Yes](https://github.com/mmalekzadeh/motion-sense/tree/master/data)           |       Has a [description](https://github.com/mmalekzadeh/motion-sense/tree/master#dataset-description)  with meta-information and [code](https://github.com/mmalekzadeh/motion-sense/tree/master#a-code-to-build-a-labeled-time-series-from-data-into-a-pandas-dataframe) to build train dataset.              |
| STS  | [Multimodal datasets](https://github.com/drmuskangarg/Multimodal-datasets)           | No           |      A list that aggregates 100+ known multimodal datasets by category              |
| STS | [MOABB](https://neurotechx.github.io/moabb/index.html) | Yes | A library that allows you to build a benchmark of popular BCI algorithms applied on the most popular BCI datasets. You can easily choose algorithms you'd like to compare, download datasets from the [list](https://neurotechx.github.io/moabb/dataset_summary.html), set corresponding [paradigms](https://neurotechx.github.io/moabb/main_concepts.html#paradigm), and use `sklearn.pipeline` to create your pipelines. An example of such end-to-end process can be found [here](https://neurotechx.github.io/moabb/auto_tutorials/tutorial_3_benchmarking_multiple_pipelines.html#sphx-glr-auto-tutorials-tutorial-3-benchmarking-multiple-pipelines-py). |

### Week 9. Topics to discuss
Prepare sheaves (ideas of the project) to discuss with slide 3 (the whole project in one slide with main notions and notations). The following four items are discussed:
1. __Title__: Title
2. __Problem__: Problem description
3. __Data__: Data and __code__ description
4. __Reference__: Links to the literature
5. __Base solution__: Description of the  basic solution to start from
6. __Proposed solution__: description of the idea to implement in the project
7. __Novelty__: why the task is good and what does it bring to science?  (for the editorial board and reviewers)

__Project 1.__
SSM. For a given set of spatial time series, one has to make a convolution transform. To transform a Hankel (or another convenient multi-linear operator) shall be used. Find the optimal number of indexes and optimal dimensionality for each index. Reconstruct the initial spatial time series.

__Project 2.__
GBM. For a given set of time series, one has to make a dynamic graph, prune it, and reconstruct the initial time series (according to MSE or another convenient criterion).

__Project 3.__
GEN2. For several (two) modalities of (spatial) time series (of various nature) construct a generative model that reveals a (probabilistic) manifold in each modality. So that there is a (time-synchronized) map between the manifolds of modalities.

__Project 4.__
GMS. The uncertainty of classification is estimated as a distance of two probability distributions. The first one is the distribution of the generative model, and the second one is the observed.

### Week 10. Topics to discuss
The same format with improvements:
1. Draw the "all project" slide using languages:
   * Category theory
   * Plate notation
2. Narrow the goal and problem statement so the plan of work is visible
3. Express one of two discussed scientific criteria:
   * find a non-investigated topic, an error, a new point-of-view
   * prove that the proposed model is adequate using statistical methods (inference with a chosen informative prior)
4. Describe new projects from the list or propose yours:

__Project 5.__
SSM. Inverse problem (in BCI or CT): probabilistic versus deterministic approach (with application to a designated dataset)

__Project 6.__
DBA. Probabilistic approach to DTW-DBA problem statement (with application to a designated dataset)

__Project 7.__
GEN2. Probabilistic criteria of manifold-to-manifold mapping reconstruction in multimodal CCA

### Week 11. Topics to discuss

__Project 8.__
Geometric (Clifford) algebra-based tools to forecast spatial time series (to continue [these papers](https://link.springer.com/journal/6) and [this paper](https://arxiv.org/abs/2302.06594))

__Project 9.__
Continuous time and space memory networks ([this is an old paper](https://proceedings.neurips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf), what is next?)

__Project 10.__
Differential geometry connects continuous-time graph Laplacian and Riemannian metric tensor (after [GRAND](https://arxiv.org/abs/2106.10934), [Dutta]( https://doi.org/10.1016/j.neuroimage.2021.118126))

__Project 11.__
(a variant of 8) Application of De Rham's theorem (Green's, Stocks' theorem) in Physics-informed machine learning ([after this paper](https://www.nature.com/articles/s41467-021-26434-1))

__Essay.__
My forecast of ML development in the next ten years (with approximate plan)
1. What trends and topics will impact scientific research 
2. What tools will be most useful in academic and industrial research
3. What applications will be the most significant
4. What is my role in the profession, how I will develop my career

## Table of contents  
### Week 1
1. [D4](week_1/D4/main.pdf)
2. [Graph Laplacian](week_1/Graph_Laplacian/Graph%20Laplacian.pdf)
3. [Variational CCA](week_1/Var_CCA/main.pdf)

### Week 2
1. [Generative Alternatives](week_2/Generative_Alternatives/Generative_alternatives.pdf)
2. [Joint VAEs with Normalizing Flows](week_2/JNF_DCCA/main.pdf)
3. [Methods comparison for graphs](week_2/Riemann_vs_Graph/Riemann_vs_Graph.pdf)
4. [Sliced-Wasserstein](week_2/SW_distance/SW_distance_SPD.pdf)

### Week 4
1. [Score-based Multimodal Autoencoders](week_4/score_ae/main.pdf)
2. [Graph Auto-regressive models](week_4/Graph_AR/Graph_AR.pdf)
3. [Time series reconstruction](week_4/ts_reconstruction/main.pdf)

### Week 8
1. [Graph Neural Networks on SPD Manifolds for Motor Imagery Classification: A Perspective From the Time–Frequency Analysis [Riemannian model]](https://github.com/intsystems/IDA/blob/main-2023/week_7/Graph-CSPNet/Graph_CSPNet.pdf)

### Week 9
1. [Reconstruction of time series using S4](week_9/SSM/S4.pdf)
2. [Reconstruction of time series using pruning via sampling](week_9/dyn_graph_reconstruction/Dynamic%20graph%20pruning%20an%20reconstruction%20.pdf)
3. [Dimension Reduction for Time Series with Score-based generative modeling](week_9/sb_dim_reduction/main.pdf)

### Week 10

