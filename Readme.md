# Intelligent data analysis

**This year main topic: Brain Computer Interface**

Brain-computer interface with Functional data analysis Human behavioral analysis and forecasting requires models that have to predict target variables of complex structures. We develop PLS and CCA (Projection to latent space and Canonic correlation analysis) methods towards the Multiview with continuous-time data representation.

## Generative time series decoding models  
* **The goal** is to create a generative state-space model for BCI
* **The impact**  is to boost the behavioral classification quality by decision-rejecting 
* **The principle** if a generated pattern does not belong to one of the expected patterns (one-class classification) we reject the decision
* **The plan**: 
  * create the simplest generative model for selected data
  * apply SSM (state-space model) principles to make CCA (canonic-correlation analysis)
  * introduce classification model and decision-rejecting criterion
  * compare quality 

## References
* Direct Discriminative Decoder Models for Analysis of High-Dimensional Dynamical Neural Data by M.R. Rezaei et al. 2022 [DOI](https://doi.org/10.1162/neco_a_01491)
* Deep Direct Discriminative Decoders for High-dimensional Time-series Data Analysis by M.R. Rezaei 2023 (NTDB)
* Decoding Hidden Cognitive States From Behavior and Physiology Using a Bayesian Approach by Ali Yousefi et al. [DOI](https://doi.org/10.1162/neco_a_01196)
* Bayesian Decoder Models with a Discriminative Observation Process by M.R. Rezaei et al. 2020 [DOI](https://doi.org/10.1101/2020.07.11.198564)
* Deep Discriminative Direct Decoders for High-dimensional Time-series Analysis by M.R. Rezaei 2020 [ArXiv](https://arxiv.org/abs/2205.10947)
* Real-Time Point Process Filter for Multidimensional Decoding Problems Using Mixture Models by M.R. Rezaei 2020 [DOI](https://doi.org/10.1016/j.jneumeth.2020.109006)
* [Basic Code D4](https://github.com/vadim-vic/Deep_Direct_Discriminative_Decoder-D4- Basic code D4]
* Variational auto-encoded deep gaussian processes by Z. Dai et al. 2021 [ArXiv](https://arxiv.org/abs/1511.06455 ArXiv]
* Parametric Gaussian process regressors by M. Jankowiak et al. 2020 [ArXiv](https://arxiv.org/abs/1910.07123 ArXiv]
* A Tutorial on Gaussian Processes by Z. Ghahramani 2010 [ArXiv](http://learning.eng.cam.ac.uk/zoubin PDF]
* An Intuitive Tutorial to Gaussian Processes Regression by J. Wang 2021 [ArXiv](https://arxiv.org/abs/2009.10862 ArXiv]

==Riemannian Geometry and Graph Laplacian metric models==

# '''The goal''' is to create a metric '''behavioral''' forecasting model for BCI
# '''The impact'''  is to construct time-embedding metric space so that it is compatible with the generative models 
# '''The principle''' a dynamic system changes its state consequently, so we construct a metric state space that could be decomposed with one of diffusion models
# '''The plan''' 
## select a metric model with continuous time
## use Riemannian geometry and Graph-Laplacian approaches
## make diffusion decomposition
## boost decoding models with metric space


===References===
# Classification of covariance matrices using a Riemannian-based kernel for BCI applications by A. Barachant et al. 2013 (Neurocomputing)
# Multi-class Brain-Computer Interface Classification by Riemannian Geometry by A. Barachant et al. 
# Riemannian Geometry for EEG-based Brain-Computer Interfaces by M. Congedo et al.
# Online SSVEP-based BCI using Riemannian geometry by E. K. Kalunga 2016 [http://dx.doi.org/10.1016/j.neucom.2016.01.007 DOI]
# A Plug&Play P300 BCI Using Information Geometry by A. Barachant 2014 [https://arxiv.org/abs/1409.0107 ArXiv]
# Longitudinal predictive modeling of tau progression along the structural connectome by J.Dutta et al. 2021 [https://doi.org/10.1016/j.neuroimage.2021.118126 DOI]
# Grand: Graph neural diffusion by M.M. Bronstein et al. ICML, 2021.
# (inspiring) The inverse problem in electroencephalography using the bidomain model of electrical activity by A.L. Rincon and S. Shimoda, 2016 [http://dx.doi.org/10.1016/j.jneumeth.2016.09.011 DOI]
# (inverse)  High-Resolution EEG Source Reconstruction with Boundary Element Fast Multipole Method, N. Makaroff et al. 2022 [https://doi.org/10.1101/2022.10.30.514418 DOI]

==Data==
Any data that has
# timeline with a behavioral pattern, synchronous both for source and target data
# source time series with 
#* probabilistic assumptions for diffusion probabilistic models
# target time series with
#* behavioral pattern to make a decision

To select from
* [[BCI|List of datasets and tools]] 
* Scientific lessons from a catalog of 6674 brain recordings by A.D.J. Makin

==The problem==
# To make a classification decision or to reject it
# To forecast a system behavior (system state) and generate variants
# The rejection criterion is a mismatch observation from generated scenarios

==Assumptions==
# Short time series (relatively, hundreds or thousands of samples)
# Time series have big variances and systematic errors
# Time series could be significantly correlated
# Time series have origins
## exogenous, no one can control
## control signals 
## and decisions
## behavioral 
# Timeline has structure
## periods (seasonal or quasi)
## events (forced or selected)
## including decision timestamps

==Examples==
# Any sports game (exogenous, behavior to model, decisions)
# Brain computer interfaces
# Risk management, stock market

==The challenge is to put==
* generative models 
* of time series graph Laplacian
* into seq2seq framework
* to make quality decisions.

==Schedule 2023==
{|class="wikitable"        
|-
| '''Date'''
| 
| '''N'''
| '''Progress'''
| '''To discuss'''
| '''Result'''
|-  
|September
|16
|1	
|Introduction and planning
|Topics for the next week 
|Subscribed to the schedule
|-  
|
|23
|2
|Possible models
|Introductions to models 
|Slides, references, questions to reason
|-
|
|30
|3
|Model reasoning and selection
|Discussion of the questions, refined presentations of models
|Role of models in the system
|-
|October
|7
|4
|System description
|Variants of the system, code sources
|List of code references, toolbox(es) selection
|-
|
|14
|5
|Architecture planning
|Variants of the pipelines
|Collection of pipelines, executable 
|-
|
|21
|6
|Data collection
|Discussion of the applications 
|Links to the data and downloaders
|-
|
|28
|7
|Minimum value system
|Discussion of the tutorials
|Tutorial plan
|-
|November
|4
|8
|Examples and documents  
|Tutorial presentation
|List of new models and challenges
|-
|
|11
|9
|New directions
|Model presentation (week 2)
|List of challenges
|-
|
|18
|10
|Analysis, testing, and applications
|Decomposition of the system
|List of possible projects
|-
|
|25
|11
|Project proposals
|Project presentations
|Project descriptions
|-
|December
|2
|12
|Project description review
|Discussion of the messages
|Published descriptions
|-
|}


=== Week 2. Topics to discuss === 
# CCA generative models 
# Continous models for graph Laplacian
# D4 and variants 
# Riemmanian model with time
# SSM generative models
# Generative models with time graph convolutions 
# (risky) RL for time series scenario generation
# (math) CCA Error functions and (tech) Autograd for seq2seq generative pipelines
