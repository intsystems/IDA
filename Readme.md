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
|            | 4   | 7   | Minimum value system                 | Discussion of the tutorials                                   | Tutorial plan                                   |
| November   | 11  | 8   | Examples and documents               | Tutorial presentation                                         | List of new models and challenges               |
|            | 18  | 9   | New directions                       | Model presentation (week 2)                                   | List of challenges                              |
|            | 25  | 10  | Analysis, testing, and applications  | Decomposition of the system                                   | List of possible projects                       |
| December   | 2   | 11  | Project proposals                    | Project presentations                                         | Project descriptions                            |
|            | 9   | 12  | Project description review           | Discussion of the messages                                    | Published descriptions                          |

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
1. Canonical correlation analysis (and generative)
2. Generative models: 1) Normalizing flows and Autoregressive models, 2) Probabilistic diffusion models and Variational autoencoders
3. Graph and dynamic barycenter models
4. State-space models 1) discrete-time, 2) continous-time
5. Cross-convergence mapping
6. Generative Bayesian model selection
7. Useful seq2seq models and transformers

| Toolbox    | Doc, Demo | Pipeline | Comment on usability |
|------------|-----------|----------|----------------------|
| [pyRiemann: Biosignals classification with Riemannian geometry](https://pyriemann.readthedocs.io/en/latest/)       | Y, Y          | sk-learn         | Quick start with useful extensive demos and data ready to go.                     |
|         |           |          |                      |
|         |           |          |                      |
|         |           |          |                      |
|         |           |          |                      |
|         |           |          |                      |
|         |           |          |                      |
|         |           |          |                      |
|         |           |          |                      |
|         |           |          |                      |
|         |           |          |                      |




 
