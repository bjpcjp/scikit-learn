
## Getting Started
## [Changes (v1.3.2)](https://scikit-learn.org/stable/whats_new/v1.3.html)
## [Dev Guide](https://scikit-learn.org/dev/developers/index.html)
### contributing
### Creating a minimal reproducer
### Building an estimator
### Tips
### Utilities
- [input validations](https://scikit-learn.org/dev/developers/utilities.html#validation-tools)

- [linear algebra & array ops](https://scikit-learn.org/dev/developers/utilities.html#efficient-linear-algebra-array-operations)

- [random sampling](https://scikit-learn.org/dev/developers/utilities.html#efficient-random-sampling)

- [sparse matrices](https://scikit-learn.org/dev/developers/utilities.html#efficient-routines-for-sparse-matrices)

- [graphs](https://scikit-learn.org/dev/developers/utilities.html#graph-routines)

- [testing](https://scikit-learn.org/dev/developers/utilities.html#testing-functions)

- [multiclass/multilabel](https://scikit-learn.org/dev/developers/utilities.html#multiclass-and-multilabel-utility-function)

- [general helpers](https://scikit-learn.org/dev/developers/utilities.html#helper-functions)

- [hash functions](https://scikit-learn.org/dev/developers/utilities.html#hash-functions)

- [warnings & exceptions](https://scikit-learn.org/dev/developers/utilities.html#warnings-and-exceptions)

### Speed optimization
- code profiling
- memory profiling
- cython
- profiling compiled extensions
- multicore ops with __joblib.Parallel__

## User Guide
### Linear Models
- [Least Squares](https://scikit-learn.org/dev/modules/linear_model.html#ordinary-least-squares)
    - Least Squares (Non-Negative)
    - Complexity
    
- Ridge
    - Regression
    - Classification
    - Complexity

- Ridge Regression w/ CV
- Lasso Regression
- Lasso Regression (Multitask)
- Elastic Net
- Elastic Net (Multitask)
- Elastic Net w/ CV
- Least Angle Regression (LARS)
- LARS Lasso
- OMP
- Bayes Regression
- General Linear Regression (GLR)
- Tweedie Regression
- Stochastic Gradient Descent (SGD)
- Perceptrons
- Passive Aggressive Algos
- RANSAC
- Huber
- Theil-Sen

### Logistic Regression (LR)
- Intro
- Cross-Validated
- Ex: Sparsity vs Penalties
- Ex: Regularization vs C
- Ex: Multinomial vs OVR
- Ex: Document classif
- Ex: Multinomial with L1, MNIST

### Discriminant Analysis
- Intro
- Ex: LDA vs PCA (Iris)
- Math Basics
- Shrinkage
- Ex: Covariance, LDA classif
- Estimators

### Kernel Ridge Regression (KRR)
- Intro
- Ex: KRR vs SVR
- Ex: KRR vs SVR, CPU time

### Support Vector Machines (SVMs)
- Intro
- Class. (SVC, NuSVC, LinearSVC)
- Ex: Max margin hyperplane plot
- Ex: Binary classif, RBF kernel
- Ex: SVM, univarite feature select
- Ex: Multiclass classif, SVC, NuSVC
- Scoring
- Weights
- Ex: SVC, Unbalanced classes  
- Ex: Weighted samples
- SVR (regression)
- Ex: SVR - linear/poly/RBF kernels
- Computation
- RBF kernel params (C, gamma)
- Custom kernels
- Precomputed kernels - Gram matrix

### Stochastic Gradient Descent (SGD)
- Intro
- SGD Classif
- SGD Classif (Multiclass)
- SGD Classif (Weighted)
- Averaging
- Ex: Solvers (SGD, AGSD, SAG,...)
- SGD Regrssn
- Sparse Data
- Complexity
- Stopping & Convergence
- Tips

### K Nearest Neighbors (KNN)
- Intro
- BallTree, KDTree, Brute Force
- Ex: Basics
- Ex: Find Sparse Graph
- Radius-Based Classif
- Ex: Uniform vs Distance Weights
- Radius-Based Regrssn
- Ex: Uniform vs Distance Weights
- Ex: Upper/Lower Face Matching
- KD Tree vs Ball Tree vs Brute Force
- Nearest Centroids
- Ex: Shrink Thresholds
- Radius-based Transformers
- Ex: Approx NNs > TSNE
- Ex: Caching
- Neighborhood Components Analysis
- Ex: NCA
- NCA Classif
- Ex: KNN Classif with/without NCA
- NCA vs LDA vs PCA

### Gaussians
- Intro
- Regrssn (GPR)
- Ex: GPR w/ noise estimate
- GPR vs KRR

### Cross Decomposition
- Intro/Partial Least Sq. (PLS)
- Canonical PLS
- SVD PLS
- PLS Regrssn (PLSR)
- Ex: PLSR / Multivariate
- Ex: PLSR / Univariate
- Canonical Correlation Analysis (CCA)
- Ex: PCR vs PLSR

### Naive Bayes (NB)
- Intro
- NB Classif (Gaussian)
- NB Classif (Multinomial)
- NB Classif (Complement)
- NB Classif (Bernoulli)
- Categorical NB
- Out-of-core NB

### Decision Trees (DTs)
- Intro
- DT Classifiers
- Graphviz
- Ex: DTs, Iris
- DT Regression
- Multi-output DTs
- Ex: Multi-output DT Rgressn
- Ex: Face Image Completion
- Complexity
- ID3, C4.5, C5.0, CART
- Impurity
- Min-Cost Pruning

### Ensembles/Bagging
- Intro
- Ex: Bagging vs 1 Estimator
- Random Forests
- Extra Trees
- Parameters
- Parallel Execution
- Rank aka Feature Importance
- Ex: Extra Trees
- Random Tree Embedding (RTE)
- Ex: RTE - Hashing features

### Ensembles/Boosting
- AdaBoost
- Ex: SAMME vs SAMME.R
- AdaBoost (Multiclass)
- Ex: 2-class Adaboost
- Ex: DT Rgrssn with AdaBoost
- Gradient Boosted DTs
- Histogram Gradient Boosting (HGB)
- Ex: HGB - Category Features
- HGB - Monotonic Constraints
- HGB Performance
- Stacked Generalization (SG)

### Ensembles/Voting
- Hard Voting Classifier
- Soft Voting Classifier
- Voting Regressor

### General Stacking
- Intro

### Multiclass, Multioutput
- Intro
- Label Binarization
- One vs Rest (OVR) Classifiers
- Multilabel Classifiers
- One vs One (OVO) Classifiers
- Output Code Classifiers
- Multioutput Classifiers
- Classifier Chains
- Multitask Classifiers
- Mutlioutput Regression
- Regressor Chaining

### Feature Selection (FS)
- Removing Low-Variance Feats
- Univariate FS
- Recursive FS
- FS from a Model
- FS from a Tree (Impurity Based)
- Sequential FS
- FS & Pipelines

### Unsupervised Learning
- Intro
- Self-Training Classifiers
- Threshold vs Self-Training
- Ex: Decision Boundaries
- Label Propagation/Spreading

### Isotonic Regression
- Intro

### Calibration Curves (api: [sklearn.calibration](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.calibration))
- Intro
- Ex: Classifier Comparison
- Calibration
- Cross Validation
- Metrics
- Regressors
- Multiclass Support
- [examples](https://scikit-learn.org/stable/auto_examples/index.html#calibration)

### Perceptrons
- Intro
- Classifiers
- Regressors
- Regularization
- Training (SGD, Momentnum, Adam, L-BFGS)
- Complexity & Tips
- Control Options

### Gaussian Mixtures
- Intro
- Expectation Maximization (EM)
- Ex: GMM Clustering
- Ex: GMM Density Estimation
- Ex: GMM & Bayes Info Criterion (BIC)
- Variational Bayes GM
- Ex: Concentration Prior Analysis
- Ex: GM Confidence Ellipsoids
- Ex: GMM, Sinusoidal Data

### Manifolds
- Intro
- Ex: Dimensionality Reduction (DR)
- Isomap
- Locally Linear Embedding (LLE)
- Modified LLE
- Hessian LLE
- Local Tangent Space Align (LTSA)
- Multi Dimensional Scaling (MDS)
- Metric/non-Metric MDS
- Random Tree Embedding
- Spectral Embedding
- t-SNE
- Neighborhood Components Analysis (NCA)

### Clustering (api: [sklearn.cluster](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster))
- Intro
- K-Means
- K-Means Assumptions vs Performance
- K-Means & Voronoi Diagrams
- K-Means (Minibatch)
- Affinity Propagation
- Mean Shift
- Spectral Clustering
- Agglomerative Clustering (AC)
- Hierarchical Clustering / Dendrograms
- AC: Connectivity Constraints
- AC: Connectivity Graphs
- AC: Distance Metrics
- DBSCAN
- OPTICS
- Birch

### Clustering Metrics
- Rand Index
- Mutual Info Score
- Homogeneity, Completeness, V-Measure
- Fowlkes-Mallows
- Silhouette Coefficient
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Contingency Matrix
- Pair Confusion Matrix

### Biclustering
- Intro
- Spectral Co-Clustering
- Spectral Bi-Clustering
- Metrics
- [examples](https://scikit-learn.org/stable/auto_examples/index.html#biclustering)

### Component Analysis / Matrix Factorization
- Principal Component Analysis (PCA)
- Probabilistic PCA vs FA
- PCA vs LDA
- Incremental PCA
- PCA with Random SVD
- Sparse PCA
- Dimension Reduction Method Comparison
- Kernel (Non-Linear) PCA
- Truncated SVD
- Document Clustering
- Dictionary Learning (DL): Sparse Coding
- DL: Image Denoising
- DL: Minibatch
- Factor Analysis (FA)
- Independent Component Analysis (ICA)
- NonNegative Matrix Factorization (NNMF)
- Latent Dirichlet Allocation (LDA)

### Covariance
- Empirical
- Shrunk
- Ledoit-Wolf (LW) Shrinkage
- Oracle Approximating (OA) Shrinkage
- Precision Matrix
- Min Covariance Determinants (MCDs)

### Novelties & Outliers
- Intro
- Method Comparison
- One Class SVM
- Elliptic Envelope
- Isolation Forest
- Local Outlier Factor (LOF)
- LOF & Novelty Detection

### Density Estimation
- Intro
- Histograms
- Kernel Density Estimation (KDE)

### Restricted Boltzmann Machines
- Intro
- Learning - Stochastic Max Likelihood (SML)

### Cross Validation
- Intro
- Metrics
- Scoring
- Examples
- KFold CV
- Stratified KFold CV
- Leave One Out (LOO)
- Leave P Out (LPO)
- Shuffle & Split
- Stratified KFold
- Stratified Shuffle Split
- Group KFold
- Leave One Group Out (LOGO)
- Leave P Groups Out (LPGO)
- Group Shuffle Split (GSS)
- Predefined Split Methods
- Time Series Split (TSS)
- Permutation Testing
- Visualizations

### Hyperparameters
- Intro
- Grid Search (GS)
- Randomized Search (RS)
- Successive Halving
- Tips
- Composite Estimators & Param Spaces
- Methods with Built-in Search
- Info Criteria (AIC, BIC)

### Metrics
- Intro / Scoring
- make_scorer
- Multiple Metrics

### Classification Metrics
- Scoring
- Precision-Recall Curve
- ROC-AUC Score
- Accuracy
- Accuracy (Top-K)
- Accuracy (Unbalanced)
- Cohen's Kappa
- Confusion Matrix
- Classification Report
- Hamming Loss
- Precision, Recall, F-measure
- Precision-Recall Curve
- Average Precision Score
- Jaccard Similarity
- Hinge Loss
- Log Loss
- Matthews Correlation Coefficient
- Confusion Matrix (Multilabel)
- Receiver Operating Characteristic (ROC)
- Detection Error Tradeoff (DET)
- Zero One Loss
- Brier Score

### Multilabel Ranking Metrics
- Coverage Error
- Label Ranking Avg Precision (LRAP)
- Label Ranking Loss
- Discounted Cumulative Gain (DCG)
- Normalized DCG

### Regression Metrics
- Explained Variance
- Max Error
- Mean Abs Error (MAE)
- Mean Squared Error (MSE)
- Mean Squared Log Error (MSLE)
- Mean Abs Pct Error (MAPE)
- R^2 Score
- Tweedie Deviances

### Dummy Metrics
- Intro & Alternatives

### Validation & Learning Curves
- Example: Underfit/Overfit
- Validation Curve
- Learning Curve

### Inspection Plots
- Partial Dependence Plots (PDPs)
- Individual Conditional Expect (ICE) Plot
- Computation

### Permutation Feature Plots
- Intro
- Impurity vs Permutations

### ROC Curves
- Examples

### Partial Dependence Plots
- Examples

### Display Objects
- Examples

### Composite Tranforms
- Pipelines
- Caching
- Regression Target Transformers
- Feature Unions
- Column Transformers
- Visualization (HTML)

### Feature Extraction (FE)
- Dicts
- Dicts and NLP
- Feature Hashing

### Text FE
- Bag of Words
- Sparsity
- Count Vectorizer
- Stop Words
- Tf-IDF
- Text File Decoding/ UTF-8
- The Hashing Trick
- Out of Core Scaling
- Custom Vectorizer Classes

### Image FE
- Extraction
- Reconstruction
- Connectivity Graphs

### Preprocessing
- Scaling
- Quantile Transforms
- Power Maps (Box-Cox, Yeo-Johnson)
- Normalization
- Ordinal Encoding
- One Hot Encoding
- Quantization aka Binning
- Feature Binarization

### Missing Values / Imputation
- Univariate
- Multivariate
- Iterative
- Multiple vs Single Imputation
- Nearest Neighbors

### Random Projections (RPs)
- Intro
- Johnson-Lindenstrauss lemma
- Gaussian RP
- Sparse RP
- Empirical Validation
- Marking Missing Values

### Kernel Approximations
- Intro
- Nystroem
- RBF Sampler
- Additive Chi-Squared Sampler
- Skewed Chi-Squared Sampler
- Polynomial Sampling / TensorSketch

### Pairwise Ops
- Distances vs Kernels
- Cosine Similarity
- Kernels: Linear, Polynomial, Sigmoid, RBF, Lapacian, Chi-Squared

### Transform Prediction Targets
- Label Binarization (LB)
- LB (Multilabel)
- Label Encoding

### Example Datasets
- Toy datasets
- Real world datasets

### Artificial Data Generators
- For classifier problems
- For multilabel classifier problems
- Hastie classifiers
- Bicluster data
- For regression problems
- For manifolds
- For decomposition

### Other Dataset APIs
- Sample Images
- SVMlight, Libsvm formats
- OpenML.org
- Pandas.io
- Scipy.io
- Numpy.routines.io
- Scikit-image
- Imageio

### Scalability
- Out of Core (OOC) Ops

### Performance
- Latency Factors
- Model Complexity
- Linear Algebra: BLAS, LAPACK
- Working Memory
- Model Compression / Sparsify
- Model Reshaping

### Parallelism
- Joblib
- OpenMP
- MKL, OpenBLAS, BLIS
- Oversubscription (of threads)
- Config switches

### File Persistence
- Pickle
- Joblib
