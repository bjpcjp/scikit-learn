### scikit-learn contents

| Section | Title | Contents |
| ------- | ----- | -------- |
| 00      | **Getting Started**                   | Estimators, Transformers, Preprocessors, Pipelines, Model Evaluation, Parameter Searches, Next Steps |
| 01      | **Linear Models**                     | OLS, Ridge, Lasso, Elastic-Net, Least Angle Regression (LARS), LARS Lasso, OMP, Naive Bayes, Generalized Linear Models (GLM), Tweedie Models, Stochastic Gradient Descent (SGD), Perceptrons, Passive-Aggressive Algos, Polynomial Regression |
| 01a     | **Logistic Regression**               | Basics, Examples |
| 01b     | **Splines**                           | Polynomial Regression & Basis Functions, Periodic splines |
| 01c     | **Quantile Regression**               | Examples, QR vs linear regression |
| 01d     | **Outliers**                          | Robustness, RANSAC, Huber, Thiel-Sen |
| 02      | **Discriminant Analysis**             | LDA, QDA, Math Foundations, Shrinkage, Estimators |
| 03      | **Kernel Ridge Regression**           | KRR vs SVR |
| 04      | **Support Vector Machines (SVMs)**    | Classifiers, Regressors, Scoring, Weights, Complexity, Kernels |
| 05      | **Stochastic Gradient Descent (SGD)** | Classifiers, Solvers, Regressors, Sparse Data; Complexity; Stopping/Convergence; Tips |
| 06      | **K Nearest Neighbors (KNN)**         | Algos (Ball Tree, KD Tree, Brute Force), Radius-based KNN, Nearest Centroid Classifiers, Caching, Neighborhood Components Analysis (NCA) |
| 07      | **Gaussian Processes (GPs)**          | Regressors, Classifiers, Kernels |
| 08      | **Cross Decomposition**               | Partial Least Squares (PLS), Canonical PLS, SVD PLS, PLS Regression, Canonical Correlation Analysis (CCA) |
| 09      | **Naive Bayes (NB)**                  | Gaussian NB, Multinomial NB, Complement NB, Bernoulli NB, Categorical NB, Out-of-core fitting |
| 10      | **Decision Trees (DTs)**              | Classifiers,  Graphviz,  Regressions,  Multiple Outputs,  Extra Trees,  Complexity, Algorithms,  Gini, Entropy, Misclassification,  Minimal cost-complexity Pruning |
| 11a     | **Ensembles/Bagging**                 | Methods,  Random Forests,  Extra Trees,  Parameters,  Parallel Execution,  Feature Importance,  Random Tree Embedding |
| 11b     | **Ensembles/Boosting**                | Gradient Boosting (GBs),  GB Classifiers,  GB Regressions,  Tree Sizes,  Loss Functions,  Shrinkage,  Subsampling,  Feature Importance,  Histogram Gradient Boosting (HGB),  HGB - Monotonic Constraints |
| 11ba    | **Ensembles/Boosting/Adaboost**       | examples |
| 11c     | **Ensembles/Voting**                  | Hard Voting,  Soft Voting,  Voting Regressor |
| 11d     | **Ensembles/General Stacking**        | Summary |
| 12      | **Multiclass/Multioutput Problems**   | Label Binarization,  One vs Rest (OvR), One vs One (OvO) Classification,  Output Codes,  Multilabel, Multioutput Classification,  Classifier Chains,  Multioutput Regressions,  Regression Chains |
| 13 | **Feature Selection (FS)**                 | Removing Low-Variance Features,  Univariate FS,  | Recursive FS,  Model-based FS,  Impurity-based FS,  Sequential FS,  Pipeline Usage |
| 14 | **Semi-Supervised**                        | Self-Training Classifier,  Label Propagation,  Label Spreading |
| 15 | **Isotonic Regression**                    | Example |
| 16 | **Calibration Curves**                     | Intro/Example,  Cross-Validation,  Metrics,  Regressors |
| 17 | **Perceptrons**                            | Intro,  Classification,  Regression,  Regularization,  Training,  Complexity,  Tips |
| 21 | **Gaussian Mixtures (GMs)**                | Expectation Maximization,  Variational Bayes GM |
| 22 | **Manifolds**                              | Isomap,  Locally Linear Embedding (LLE),  Modified LLE, Hessian LLE,  Local Tangent Space Alignment (LTSA),  Multidimensional Scaling (MDS),  Random Trees Embedding,  Spectral Embedding,  t-SNE,  Neighborhood Components Analysis (NCA) |
| 23 | **Clustering**                             | K-Means,  Voronoi Diagrams,  Affinity Propagation,  Mean Shift,  Spectral Clustering,  Agglomerative Clustering,  Dendrograms,  Connectivity Constraints,  Distance Metrics,  DBSCAN,  Optics,  Birch |
| 23a | **Clustering Metrics**                    | Rand Index,  Mutual Info Score,  Homogeneity,  Completeness,  V-Measure,  Fowlkes-Mallows,  Silhouette Coefficient,  Calinski-Harabasz,  Davies-Bouldin,  Contingency Matrix,  Pair Confusion Matrix |
| 24 | **Biclustering**                           | Spectral Co-Clustering,  Spectral Bi-Clustering,  Metrics |
| 25 | **Component Analysis / Matrix Factorization** | PCA,  Incremental PCA,  PCA w/ Random SVD,  PCA w/ Sparse Data,  Kernel PCA,  Dimension Reduction Comparison,  Truncated SVD / LSA,  Dictionary Learning,  Factor Analysis,  Independent Component Analysis,  Non-Negative Matrix Factorization (NNMF),  Latent Dirichlet Allocation (LDA) |
| 26 | **Covariance**                             | Empirical CV,  Shrunk CV,  Max Likelihood Estimation (MLE),  Ledoit-Wolf Shrinkage,  Oracle Approximating Shrinkage,  Sparse Inverse CV, aka Precision Matrix,  Mahalanobis Distance |
| 27 | **Novelties & Outliers**                   | One-Class SVMs,  Elliptic Envelope,  Isolation Forest,  Local Outlier Factor |
| 28 | **Density Estimation (DE)**                | Histograms,  Kernel DE |
| 29 | **Restricted Boltzmann Machines (RBMs)**   | Intro,  Training |
| 31 | **Cross Validation (CV)**                  | Intro,  Metrics,  Parameter Estimation,  Pipelines,  Prediction Plots,  Nesting,  K-Fold,  Stratified K-Fold,  Leave One Out,  Leave P Out,  Class Label CV,  Grouped Data CV,  Predefined Splits,  Time Series Splits,  Permutation Testing,  Visualizations |
| 32 | **Parameter Tuning**                       | Grid Search,  Randomized Optimization,  Successive Halving,  Composite Estimators & Parameter Spaces,  Alternative to Brute Force,  Info Criteria (AIC, BIC) |
| 33 | **Metrics & Scoring (Intro)**              | scoring,  make_scorer | 
| 33a | **Classification Metrics**                | Accuracy,  Top-K Accuracy,  Balanced Accuracy,  Cohen's Kappa,  Confusion Matrix,  Classification Report,  Hamming Loss,  Precision,  Recall,  F-Measure,  Precision-Recall Curve,  Average Precision,  Jaccard Similarity,  Hinge Loss,  Log Loss,  Matthews Correlation Coefficient,  Receiver Operating Characteristic (ROC) Curves,  ROC-AUC,  Detection Error Tradeoff (DET),  Zero One Loss,  Brier Score |
| 33b | **Multilabel Ranking Metrics**            | Coverage Error,  Label Ranking Avg Precision (LRAP),  Label Ranking Loss,  Discounted Cumulative Gain (DCG), Normalized DCG |
| 33c | **Regression Metrics**                    | Explained Variance,  Max Error,  Mean Absolute Error (MAE),  Mean Squared Error (MSE),  Mean Squared Log Error (MSLE),  Mean Absolute Pct Error (MAPE),  R^2 score,  aka Coefficient of Determination ,  Tweedie Deviances |
| 33d | **Dummy Metrics**                         | Dummy Classifiers,  Dummy Regressors | 
| 34 | **Viz/Validation**                         | Validation Curve,  Learning Curve | 
| 41 | **Viz/Inspection**                         | 2D PDPs, 3D PDPs,  Individual Conditional Expectation (ICE) Plot |
| 42 | **Viz/Permutations**                       | Permutation Feature Importance (PFI),  Impurity vs Permutation Metrics | 
| 50a | **Viz/ROC Curves**                        | ROC Curve |
| 50b | **Viz/custom PDP Plots**                  | Example |
| 50c | **Vis/Classification metrics**            | Confusion Matrix,  ROC Curve,  Precision-Recall Curve | 
| 61 | **Composite Transformers**                 | Pipelines,  Caching,  Regression Target xforms,  Feature Unions,  Column Transformers |
| 62a | **Text Feature Extraction**               | Bag of Words (BoW),  Sparsity,  Count Vectorizer,  Stop Words,  Tf-Idf,  Binary Markers,  Text file decoding,  Hashing Trick,  Out-of-core Scaling,  Custom Vectorizers |
| 62b | **Image Patch Extraction**                | Extract from Patches,  Reconstruct from Patches,  Connectivity Graphs |
| 63 | **Data Preprocessing**                     | Scaling,  Quantile Transforms,  Power Maps (Box-Cox, Yeo-Johnson),  Category Coding,  One-Hot Coding,  Quantization aka Binning,  Feature Binarization |
| 64 | **Missing Value Imputation**               | Univariate,  Multivariate,  Multiple-vs-Single,  Nearest-Neighbors,  Marking Imputed Values | 
| 66 | **Random Projections**                     | Johnson-Lindenstrauss lemma,  Gaussian RP,  Sparse RP Empirical Validation |
| 67 | **Kernel Approximations**                  | Nystroem,  RBF Sampler,  Additive Chi-Squared Sampler,  Skewed Chi-Squared Sampler,  Polynomial Sampling - Tensor Sketch |
| 68 | **Pairwise Ops**                           | Distances vs Kernels,  Cosine Similarity,  Kernels |
| 69 | **Transforming Prediction Targets**        | Label Binarization,  Multilabel Binarization,  Label Encoding |
| 71 | **Toy Datasets**                           | Boston,  Iris,  Diabetes,  Digits,  Linnerud,  Wine,  Breast Cancer,  Olivetti faces,  20 newsgroups,  Labeled faces,  Forest covertypes,  Reuters corpus,  KDD,  Cal housing |
| 73 | **Artificial Data**                        | random-nclass-data,  Gaussian blobs,  Gaussian quantiles,  Circles,  Moons,  Multilabel class data,  Hastie data,  BiClusters,  Checkerboards,  Regression,  Friedman1/2/3,  S-Curve,  Swiss Roll,  Low-Rank Matrix,  Sparse Coded Signal,  Sparse Symmetric Positive Definite (SPD) Matrix | 
| 74 | **Other Data**                             | Sample images,  SVMlight/LibSVM formats,  OpenML,  pandas.io,  scipy.io,  numpy.routines.io,  scikit-image,  imageio,  scipy.io.wavfile |
| 81 | **Scaling**                                | Out-of-core ops (**BUG = TODO**) | 
| 82 | **Latency**                                | Bulk-vs-atomic ops,  Latency vs Validation,  Latency vs #Features,  Latency vs Datatype,  Latency vs Feature Extraction,  Linear Algebra Libs (BLAS, LAPACK, ATLAS, OpenBLAS, MKL, vecLib) |
| 83 | **Parallelism**                            | JobLib,  OpenMP,  NumPy,  Oversubscription,  config switches |
| 90 | **Persistence**                            | Pickle,  Joblib |
